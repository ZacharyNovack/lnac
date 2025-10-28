"""
T5-based seq2seq for stereo audio channel prediction (L -> R) using PyTorch Lightning
-------------------------------------------------------------------------------

File: t5_audio_seq2seq_lightning.py

What this provides
- A PyTorch LightningDataModule that loads stereo WAVs and samples random chunks of fixed length
  (reads 16-bit PCM using soundfile and maps int16 -> uint16 token ids: token = sample + 32768).
- A LightningModule that instantiates a T5 model (from Hugging Face), resizes embeddings to a
  ``vocab_size`` (default 65536 for 16-bit) and trains it autoregressively to predict the target channel.
- Per-index loss / bpb computation and optional logging of the per-index arrays to Weights & Biases.
- Logging of the last-P% bpb/loss as requested (last_p is configurable, default 0.5).
- A small example `if __name__ == '__main__'` entry to run training with WandB logging.

IMPORTANT PRACTICAL NOTES (READ THIS):
- Sequence length vs model capacity: mapping every raw audio sample to a token produces very long
  sequences (e.g. 44.1k samples/s). Transformer architectures like T5 scale O(L^2) in memory and
  will quickly become infeasible for large L. Choose chunk lengths (seq_len) carefully (typical
  starting points: 256, 512, 1024). If you need much longer context, consider a hierarchical
  approach (frame & quantize, convolutional front-end, or an encoder that downsamples), or use
  specialized long-context models.

- Vocab size = 65536 is large (mapping 16-bit samples directly). This makes the final linear + softmax
  large (vocab * hidden_dim). Consider using mu-law / 8-bit quantization (256 tokens) or VQ
  to reduce the vocabulary if you run into memory problems or slow training.

- This script uses pretrained `t5-small` weights then *resizes* the embeddings to `vocab_size`.
  That is a reasonable starting point; but note that token semantics are arbitrary (we're not using a
  linguistic tokenizer), so pretrained weights are only useful for general parameter priors.

Dependencies
------------
- python >= 3.8
- torch
- pytorch_lightning
- transformers
- soundfile
- wandb (optional but recommended for the logging features in this file)

Install example:
    pip install torch pytorch-lightning transformers soundfile wandb


Usage example
-------------
python train.py \
    --data_dir /graft4/datasets/pnlong/lnac/sashimi/data/musdb18stereo/ \
    --chunk_size 1024 \
    --batch_size 8 \
    --project t5-lnac


Caveat: this implementation logs per-index arrays to Weights & Biases as Python lists. Avoid logging
huge arrays every step — by default we log per-index only every `log_per_index_every` training steps
and once per validation epoch. Tune `log_per_index_every` to balance observability vs overhead.

"""

import os
import glob
import math
import random
import json
import io
import argparse
from typing import List, Optional
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import soundfile as sf
import wandb

import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from icecream import ic

from transformers import T5Config, T5ForConditionalGeneration, GPT2LMHeadModel

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

def minmax_scale(tensor, range_min=0, range_max=1):
    """
    Min-max scaling to [0, 1].
    """
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return range_min + (range_max - range_min) * (tensor - min_val) / (max_val - min_val + 1e-6)

def quantize(samples, bits=8, epsilon=0.01):
    """
    Linearly quantize a signal in [0, 1] to a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long()

def dequantize(samples, bits=8):
    """
    Dequantize a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    return samples.float() / (q_levels / 2) - 1

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor((1 << bits) - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = minmax_scale(audio, range_min=-1, range_max=1)

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Shift signal to [0, 1]
    encoded = (encoded + 1) / 2

    # Quantize signal to the specified number of levels.
    return quantize(encoded, bits=bits)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = (1 << bits) - 1
    # Invert the quantization
    x = dequantize(encoded, bits=bits)

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu

    # Returned values in range [-1, 1]
    return x

def linear_encode(samples, bits=8):
    """
    Perform scaling and linear quantization.
    """
    samples = samples.clone()
    samples = minmax_scale(samples)
    return quantize(samples, bits=bits)

def linear_decode(samples, bits=8):
    """
    Invert the linear quantization.
    """
    return dequantize(samples, bits=bits)

def q_zero(bits=8):
    """
    The quantized level of the 0.0 value.
    """
    return 1 << (bits - 1)


# ----------------------------- Dataset ---------------------------------
class StereoWavChunkDataset(Dataset):
    """Dataset that yields random fixed-length chunks (frames) from stereo WAV files.

    Each item is a dict with 'input_ids' and 'labels', both LongTensors of shape (chunk_size,).
    Input = left channel, labels = right channel, both mapped from int16 -> uint16 (0..65535)
    by adding 32768.
    """

    def __init__(self, file_list: List[str], chunk_size: int, sample_rate: int = 44100, encoder_context_frames: int = 0, enc_dropout: float = 0.5, decoder_only=False, mono=False):
        self.file_list = file_list
        self.chunk_size = int(chunk_size)
        self.encoder_context_frames = encoder_context_frames
        self.sample_rate = sample_rate
        self.quantizer = 'linear'  # or 'mu-law'
        # quickly iterate over all the files to get lengths of each file
        if len(self.file_list) == 0:
            raise ValueError("file_list is empty")
        print(f"StereoWavChunkDataset: {len(self.file_list)} files, chunk_size={self.chunk_size}")
        lengths = json.load(open('musdbstereo_lengths.json', 'r'))
        for ix, f in enumerate(tqdm(self.file_list)):
            self.file_list[ix] = (f, lengths[os.path.basename(f)])  # (path, num_samples)
        self.enc_dropout = enc_dropout
        self.decoder_only = decoder_only
        self.mono = mono
            
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, file_length = self.file_list[idx]
        # randomly sample a chunk of chunk_size from the file
        chunk_size = self.chunk_size + self.encoder_context_frames*2
        offset = torch.randint(0, max(1, file_length - chunk_size), (1,)).item()
        aud, sr = torchaudio.load(path, normalize=False, frame_offset=offset, num_frames=chunk_size, backend="soundfile")
        # if not 16 bit PCM, convert to 16 bit PCM
        if aud.dtype != torch.int16:
            aud = linear_encode(aud, bits=16)
        else:
            aud = aud.long() + 32768  # map int16 -> uint16

        if aud.shape[1] < self.chunk_size:
            # pad if too short
            pad_width = self.chunk_size - aud.shape[1]
            aud = F.pad(aud, (0, pad_width), mode='constant', value=q_zero(bits=16))
        
        if aud.shape[0] < 2:
            # duplicate if mono
            aud = aud.repeat(2, 1)
        elif aud.shape[0] > 2:
            # take first two channels if more than 2
            aud = aud[:2, :]

        # convert to uint16 token ids
        left = aud[0, :].long()  # shape (chunk_size,)
        right = aud[1, :].long()  # shape (chunk_size,)

        # randomly sample to make left or right the target
        if torch.rand(1).item() < 0.5:
            input_ids = right
            labels = left
            order = 'R->L'
        else:
            input_ids = left
            labels = right
            order = 'L->R'

        # optionally apply encoder input dropout (randomly zero out all frames in input_ids)
        if (not self.decoder_only and self.enc_dropout > 0.0 and torch.rand(1).item() < self.enc_dropout) or self.mono:
            input_ids = torch.zeros_like(input_ids)

        if self.encoder_context_frames > 0:
            # input ids gets the center chunk plus encoder_context_frames on each side
            # labels gets only the center chunk, so we need to trim labels
            labels = labels[self.encoder_context_frames:-self.encoder_context_frames]
        # print(input_ids.max(), input_ids.min(), labels.max(), labels.min())
        if self.decoder_only:
            # concatenate input and labels
            if not self.mono:
                # reduce length of both sequences by half and concatenate
                if not (self.enc_dropout > 0.0 and torch.rand(1).item() < self.enc_dropout):
                    input_ids = input_ids[: self.chunk_size // 2]
                    labels = labels[: self.chunk_size // 2]
                    labels = torch.cat([input_ids, labels], dim=0)
            return {'labels': labels, 'path': path, 'order': order}
        return {
            'input_ids': input_ids,  # shape (chunk_size,)
            'labels': labels,
            'path': path,
            'order': order,
        }


# --------------------------- DataModule --------------------------------
class StereoDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 chunk_size: int = 2048,
                 encoder_context_frames: int = 0,
                 enc_dropout: float = 0.5,
                 decoder_only: bool = False,
                 mono: bool = False,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 seed: int = 42,
                 sample_rate: int = 44100):
        super().__init__()
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.encoder_context_frames = encoder_context_frames
        self.enc_dropout = enc_dropout
        self.decoder_only = decoder_only
        self.mono = mono
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.sample_rate = sample_rate

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # nothing to download
        pass

    def setup(self, stage: Optional[str] = None):
        # find wav files
        exts = ['wav', 'flac', 'aiff', 'aif']
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(self.data_dir, f'**/*.{ext}'), recursive=True)
        files = sorted(files)
        if len(files) == 0:
            raise ValueError(f"No audio files found in {self.data_dir}")

        # deterministic split
        rng = random.Random(self.seed)
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, int(n * self.test_split))
        n_val = max(1, int(n * self.val_split))
        n_train = n - n_val - n_test
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        self.train_dataset = StereoWavChunkDataset(train_files, chunk_size=self.chunk_size, sample_rate=self.sample_rate, encoder_context_frames=self.encoder_context_frames, enc_dropout=self.enc_dropout, decoder_only=self.decoder_only, mono=self.mono)
        self.val_dataset = StereoWavChunkDataset(val_files, chunk_size=self.chunk_size, sample_rate=self.sample_rate, encoder_context_frames=self.encoder_context_frames, enc_dropout=self.enc_dropout, decoder_only=self.decoder_only, mono=self.mono)
        self.test_dataset = StereoWavChunkDataset(test_files, chunk_size=self.chunk_size, sample_rate=self.sample_rate, encoder_context_frames=self.encoder_context_frames, enc_dropout=self.enc_dropout, decoder_only=self.decoder_only, mono=self.mono)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ------------------------- LightningModule ------------------------------
class T5AudioModule(pl.LightningModule):
    def __init__(self,
                 pretrained_model: str = 't5-small',
                 vocab_size: int = 65536,
                 lr: float = 1e-4,
                 last_p: float = 0.5,
                 log_per_index_every: int = 500,
                 sample_rate: int = 44100,
                 weight_decay: float = 0.0,
                 warmup_steps: int = 500,
                 compile: bool = False,
                 use_lora: bool = False,
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.0,
                 lora_target_modules: Optional[List[str]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size + 1 # sos
        self.lr = lr
        self.last_p = last_p
        self.log_per_index_every = log_per_index_every
        self.sample_rate = sample_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.val_log_ctr = 0
        self.pretrained_model = pretrained_model

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ['wi', 'wo', 'lm_head', 'embed_tokens']

        # Load a small T5 and resize the token embeddings for integer tokens.
        # This will preserve pretrained weights (where possible) and expand the token embedding matrix.
        print(f"Loading pretrained T5 model '{pretrained_model}' and resizing vocab to {self.vocab_size}")
        if 't5' in pretrained_model.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        elif 'gpt2' in pretrained_model.lower():
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        if compile and hasattr(torch, 'compile'):
            print("Compiling the model with torch.compile() (PyTorch 2.0+)")
            self.model = torch.compile(self.model)
        self.model.resize_token_embeddings(self.vocab_size)

        if self.use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError('PEFT library is required for LoRA support. Install `pip install peft`.')


            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias='none',
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            print(f"Applying LoRA with config: r={self.lora_r}, alpha={self.lora_alpha}, targets={self.lora_target_modules}")
            self.model = get_peft_model(self.model, lora_config)

        if getattr(self.model.config, 'decoder_start_token_id', None) is None and 't5' in pretrained_model.lower():
            self.model.config.decoder_start_token_id = vocab_size

    def forward(self, input_ids, labels=None, decoder_input_ids=None):
        return self.model(input_ids=input_ids, labels=labels, decoder_input_ids=decoder_input_ids, return_dict=True)

    def compute_per_index_losses(self, logits, labels):
        # logits: (B, L, V), labels: (B, L)
        B, L, V = logits.shape
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)
        # ic(logits_flat.shape, labels_flat.shape)
        loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss_per_pos = loss_flat.view(B, L)

        per_index_loss = loss_per_pos.mean(dim=0)  # mean across batch -> shape (L,)
        avg_loss = per_index_loss.mean()

        # convert nats -> bits-per-byte (token is 2 bytes for 16-bit samples)
        per_index_bpb = per_index_loss / math.log(2.0)
        avg_bpb = per_index_bpb.mean()
        return avg_loss, per_index_loss, avg_bpb, per_index_bpb

    def training_step(self, batch, batch_idx):
        self.model.train()
        input_ids = batch['input_ids'].to(self.device)  if 'input_ids' in batch else None
        labels = batch['labels'].to(self.device)

        # prepare decoder inputs (T5 helper)
        if input_ids is not None:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
        else:
            # padd with start of sequence token
            inputs = torch.cat([torch.full((labels.shape[0], 1), self.vocab_size - 1, dtype=labels.dtype, device=labels.device), labels[:, :-1]], dim=1)
            outputs = self.model(inputs, return_dict=True)
            # labels = labels[:, 1:]
        logits = outputs.logits  # (B, L, V)
        if logits.shape[1] != labels.shape[1]:
            labels = labels[:, :logits.shape[1]]

        avg_loss, per_index_loss, avg_bpb, per_index_bpb = self.compute_per_index_losses(logits, labels)

        # main scalars
        self.log('train/loss', avg_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/bpb', avg_bpb, on_step=True, on_epoch=True, prog_bar=True)

        # last-P% logging (scalar)
        L = per_index_loss.shape[0]
        p = float(self.last_p)
        start_idx = int(L * (1.0 - p))
        last_p_bpb = per_index_bpb[start_idx:].mean()
        self.log('train/last_p_bpb', last_p_bpb, on_step=True, on_epoch=True, prog_bar=True)

        # occasionally log the whole per-index arrays to WandB
        step = int(self.global_step if hasattr(self, 'global_step') else 0)
        if step % max(1, self.log_per_index_every) == 0:
            self._maybe_log_arrays_to_wandb('train', per_index_loss.detach().cpu().numpy(), per_index_bpb.detach().cpu().numpy(), step)

        return avg_loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if getattr(self, 'validation_outputs', None) is None:
            self.validation_outputs = []
        input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
        labels = batch['labels'].to(self.device)

        if input_ids is not None:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
        else:
            outputs = self.model(labels, return_dict=True)
        logits = outputs.logits

        avg_loss, per_index_loss, avg_bpb, per_index_bpb = self.compute_per_index_losses(logits, labels)

        # now do the same, but flip the input and target channels
        if input_ids is not None:
            input_ids_flipped = labels
            labels_flipped = input_ids
            decoder_input_ids_flipped = self.model.prepare_decoder_input_ids_from_labels(labels_flipped)
            outputs_flipped = self.model(input_ids=input_ids_flipped, decoder_input_ids=decoder_input_ids_flipped, return_dict=True)
            logits_flipped = outputs_flipped.logits
            avg_loss_f, per_index_loss_f, avg_bpb_f, per_index_bpb_f = self.compute_per_index_losses(logits_flipped, labels_flipped)
        else:
            avg_loss_f, per_index_loss_f, avg_bpb_f, per_index_bpb_f = avg_loss, per_index_loss, avg_bpb, per_index_bpb
        min_avg_loss = min(avg_loss, avg_loss_f)
        min_avg_bpb = min(avg_bpb, avg_bpb_f)

        # return arrays for epoch aggregation
        self.validation_outputs.append({
            'per_index_loss': per_index_loss.detach().cpu(),
            'per_index_bpb': per_index_bpb.detach().cpu(),
            'min_avg_loss': min_avg_loss.detach().cpu(),
            'min_avg_bpb': min_avg_bpb.detach().cpu(),
        })
        return {
            'avg_loss': avg_loss.detach().cpu(),
            'avg_bpb': avg_bpb.detach().cpu(),
            'per_index_loss': per_index_loss.detach().cpu(),
            'per_index_bpb': per_index_bpb.detach().cpu(),
        }

    def on_validation_epoch_end(self):
        # outputs is a list of dicts for each batch. We want to average per_index arrays across batches.
        if len(self.validation_outputs) == 0:
            return
        per_index_losses = [o['per_index_loss'].numpy() for o in self.validation_outputs]
        per_index_bpbs = [o['per_index_bpb'].numpy() for o in self.validation_outputs]

        # stack and mean
        mean_per_index_loss = np.stack(per_index_losses, axis=0).mean(axis=0)
        mean_per_index_bpb = np.stack(per_index_bpbs, axis=0).mean(axis=0)

        epoch = int(self.current_epoch if hasattr(self, 'current_epoch') else 0)
        

        # log to WandB / logger
        if self.val_log_ctr % 10 == 0:
            print(f"Validation epoch {self.global_step}: logging per-index arrays to WandB / logger")
            self._maybe_log_arrays_to_wandb('val', mean_per_index_loss, mean_per_index_bpb, step=self.global_step)
            self.val_log_ctr = 0
        self.val_log_ctr += 1

        # log scalar summaries (mean across sequence)
        mean_bpb = float(mean_per_index_bpb.mean())
        self.log('val/bpb', mean_bpb, on_epoch=True, prog_bar=True)

        # log last-P% on validation set
        L = mean_per_index_bpb.shape[0]
        start_idx = int(L * (1.0 - float(self.last_p)))
        last_p_bpb = float(mean_per_index_bpb[start_idx:].mean())
        self.log('val/last_p_bpb', last_p_bpb, on_epoch=True, prog_bar=True)
        

        # log loss/bit-per-byte
        mean_loss = float(mean_per_index_loss.mean())
        self.log('val/loss', mean_loss, on_epoch=True, prog_bar=True)

        # log min of the two directions
        min_avg_bpbs = [t['min_avg_bpb'].item() for t in self.validation_outputs if 'min_avg_bpb' in t]
        min_avg_losses = [t['min_avg_loss'].item() for t in self.validation_outputs if 'min_avg_loss' in t]
        # log average min bpb/loss across batches
        if len(min_avg_bpbs) > 0:
            mean_min_bpb = float(np.mean(min_avg_bpbs))
            self.log('val/min_dir_bpb', mean_min_bpb, on_epoch=True, prog_bar=True)
        if len(min_avg_losses) > 0:
            mean_min_loss = float(np.mean(min_avg_losses))
            self.log('val/min_dir_loss', mean_min_loss, on_epoch=True, prog_bar=True)


        self.validation_outputs = []

    def _maybe_log_arrays_to_wandb(self, stage: str, per_index_loss_arr: np.ndarray, per_index_bpb_arr: np.ndarray, step: int):
        # Only log to WandB if available as logger
        try:
            if isinstance(self.logger, WandbLogger) or hasattr(self.logger, 'experiment'):
                exp = self.logger.experiment
                # some Lightning WandbLogger expose experiment as wandb module/object
                # try:
                #     exp.log({f'{stage}/per_index_loss': per_index_loss_arr.tolist(),
                #              f'{stage}/per_index_bpb': per_index_bpb_arr.tolist()}, step=step)
                # except Exception:
                #     # Last-resort: try to save as plain scalars (first N elements)
                #     small = {f'{stage}/per_index_loss_first20': per_index_loss_arr[:20].tolist(),
                #              f'{stage}/per_index_bpb_first20': per_index_bpb_arr[:20].tolist()}
                #     exp.log(small, step=step)
                # plot it as a line plot and log the image
                fig = Figure(figsize=(4.145, 8.29), dpi=100, tight_layout=True)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(2, 1, 1)
                ax.plot(per_index_loss_arr, label='Per-index Loss', color='C0')
                ax.set_title(f'{stage} Per-index Loss')
                ax.set_xlabel('Index (sample)')
                ax.set_ylabel('Loss (nats)')
                ax.grid(True)
                ax = fig.add_subplot(2, 1, 2)
                ax.plot(per_index_bpb_arr, label='Per-index Bits-per-byte', color='C1')
                ax.set_title(f'{stage} Per-index Bits-per-byte')
                ax.set_xlabel('Index (sample)')
                ax.set_ylabel('Bits-per-byte')
                ax.grid(True)
                plt.tight_layout()
                canvas.draw()
                rgba = np.asarray(canvas.buffer_rgba())
                im = Image.fromarray(rgba)
                exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)


            else:
                # Not a WandB logger: as a fallback, log first few per-index values to Lightning logs
                for i in range(min(10, len(per_index_loss_arr))):
                    self.log(f'{stage}/per_index_loss_idx_{i}', float(per_index_loss_arr[i]), on_epoch=True)
        except Exception as ex:
            print("_maybe_log_arrays_to_wandb failed:", ex)

    def configure_optimizers(self):
        # simple AdamW
        no_decay = [p for n, p in self.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # def lr_lambda(current_step: int):
        #         if current_step < self.warmup_steps:
        #             return float(current_step) / float(max(1, self.warmup_steps))
        #         return 1.0  # Or apply a different schedule after warmup

        # scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer


# --------------------------- run / CLI ---------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=2048, help='chunk length in samples (not seconds)')
    parser.add_argument('--encoder_context_frames', type=int, default=0, help='number of context frames on each side for the encoder input')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--project', type=str, default='t5-audio')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=65536)
    parser.add_argument('--pretrained_model', type=str, default='t5-small')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--last_p', type=float, default=0.5)
    parser.add_argument('--log_per_index_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate for gradient')
    parser.add_argument('--enc_dropout', type=float, default=0.5, help='encoder input dropout rate')
    parser.add_argument('--decoder_only', action='store_true', help='Use decoder-only model (GPT2) instead of encoder-decoder (T5)')
    parser.add_argument('--mono', action='store_true', help='Train on mono audio (zero out encoder input)')
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA (PEFT) adapters')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, help='Comma-separated target module name substrings for LoRA', default='wi,wo,lm_head,embed_tokens,q,k,v,o')
    return parser.parse_args()


def main():
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",          # what metric to monitor
        mode="min",                  # lower is better
        save_top_k=1,                # keep best 3 checkpoints
        save_last=True,              # always save the last epoch
        every_n_epochs=10,
        filename="t5audio-{epoch:02d}"
    )

    args = parse_args()
    pl.seed_everything(args.seed)

    dm = StereoDataModule(data_dir=args.data_dir,
                          chunk_size=args.chunk_size,
                          encoder_context_frames=args.encoder_context_frames,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          enc_dropout=args.enc_dropout,
                          decoder_only=args.decoder_only,
                          mono=args.mono)
    dm.setup()
    lora_targets = [s.strip() for s in args.lora_target_modules.split(',')] if args.use_lora else None
    model = T5AudioModule(pretrained_model=args.pretrained_model,
                          vocab_size=args.vocab_size,
                          lr=args.lr,
                          last_p=args.last_p,
                          log_per_index_every=args.log_per_index_every,
                          use_lora=args.use_lora,
                          lora_r=args.lora_r,
                          lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout,
                          lora_target_modules=lora_targets)

    # WandB logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name) if 'WANDB_API_KEY' in os.environ or True else None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else None,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_grad_batches if hasattr(args, 'accumulate_grad_batches') else 1,
        callbacks=[checkpoint_callback],
    )

    if args.use_lora and _PEFT_AVAILABLE:
        total, trainable = 0, 0
        for n, p in model.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        print(f"Parameters: total={total:,}, trainable={trainable:,} ({100.0 * trainable / total:.4f}%)")

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()