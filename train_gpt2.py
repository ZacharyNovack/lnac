import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchaudio
import numpy as np
import wandb
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Config, get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import random
import json
from tqdm import tqdm
from typing import Optional, Tuple, Union, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

from splus import SPlus

def minmax_scale(tensor, range_min=0, range_max=1):
    """
    Min-max scaling to [0, 1].
    """
    min_val = torch.amin(tensor, dim=(0, 1), keepdim=True)
    max_val = torch.amax(tensor, dim=(0, 1), keepdim=True)
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


def msb_bits_2_vocab_size(n_bits: int) -> int:
    return (1 << n_bits) + (1 << (16 - n_bits))

def quantize_unsigned_pcm_torch(x: torch.tensor, n_bits: int, kind='linear') -> torch.tensor:
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.min() < -1 or x.max() > 1:
        raise ValueError("x must be in range [-1, 1]")
    if n_bits < 1 or n_bits > 64:
        raise ValueError("n_bits must be between 1 and 64")

    # Map from [-1, 1] to [0, 1)
    if kind == 'linear':
        x_normalized = (x + 1) / 2

        # Scale by 2^n_bits (not 2^n_bits - 1) to maintain MSB invariance
        scale = 2 ** n_bits
        x_scaled = x_normalized * scale

        # Use floor to convert to integer (not round, to maintain MSB invariance)
        x_floored = torch.floor(x_scaled)

        # Clamp to valid range [0, 2^n_bits - 1]
        max_val = (2 ** n_bits) - 1
        x_clamped = torch.clip(x_floored, 0, max_val)
    elif kind == 'mu-law':
        x_clamped = mu_law_encode(x, bits=n_bits)

    return x_clamped.to(torch.int64)



def msb_torch(x: torch.tensor, orig_n_bits: int, n_bits: int) -> torch.tensor:
    # if x.dtype != torch.uint64:
    #     raise ValueError("x must be uint64")
    return (x >> (orig_n_bits - n_bits)) & ((1 << n_bits) - 1)

def lsb_torch(x: torch.tensor, n_bits: int) -> torch.tensor:
    # if x.dtype != torch.uint64:
    #     raise ValueError("x must be uint64")
    return x & ((1 << n_bits) - 1)


class TriloByteDataset(Dataset):

    def __init__(self, data_dir: str, chunk_size: int = 4096, sample_rate: int = 44100, epoch_expansion_factor: int = 10, stereo_interleave: bool = False, lb_dropout: Union[float, List[float]] = 0.0, max_bit_depth: Union[int, str] = None, metadata_path: Optional[str] = None, encoding: str = 'linear', bit_depth_cap: Optional[int] = None):
        lengths = json.load(open(metadata_path, 'r'))
        self.files = sorted(os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if (f.endswith('.flac') or f.endswith('.wav')))
        # filter for files present in lengths
        self.files = [f for f in self.files if os.path.basename(f) in lengths]
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.encoding = encoding
        if type(max_bit_depth) == str:
            self.msb_n_bits = int(max_bit_depth.split('_')[1])
            self.max_bit_depth = int(max_bit_depth.split('_')[0])
        else:
            self.msb_n_bits = 8
            self.max_bit_depth = max_bit_depth

        self.epoch_expansion_factor = epoch_expansion_factor
        self.stereo_interleave = stereo_interleave
        self.lb_dropout = lb_dropout
        self.bit_depth_cap = bit_depth_cap

        # define mask token for lower bits dropout
        match self.max_bit_depth:
            case 24 | None:
                self.lb_mask_token = 768
            case 16:
                self.lb_mask_token = msb_bits_2_vocab_size(self.msb_n_bits)
            case 8:
                self.lb_mask_token = msb_bits_2_vocab_size(self.msb_n_bits)

        if len(self.files) == 0:
            raise ValueError("files is empty")
        print(f"TriloByteDataset: {len(self.files)} files, chunk_size={self.chunk_size}")
        # pth = metadata_path.split('.')[0] + '_train.json' if 'train' in data_dir else metadata_path.split('.')[0] + '_valid.json' if 'valid' in data_dir else metadata_path
        
        for ix, f in enumerate(tqdm(self.files)):
            self.files[ix] = (f, lengths[os.path.basename(f)])  # (path, num_samples)

        # filter files for samples < chunk_size*10
        filter_rate = self.sample_rate if self.sample_rate is not None else 44100
        self.files = [file for file in self.files if file[1]['length'] >= self.chunk_size * file[1]['sample_rate'] // filter_rate * 10]
        if self.max_bit_depth is not None and self.lb_dropout == 0.0:
            print(f"Filtering files for max bit depth {self.max_bit_depth} with no lower bit dropout")
            # filter for files with bits_per_sample == max_bit_depth
            self.files = [file for file in self.files if file[1]['bits_per_sample'] == self.max_bit_depth]
        self.files = self.files * self.epoch_expansion_factor
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path, metadata = self.files[idx]
            file_length = metadata['length']
            sample_rate = metadata['sample_rate']
            # randomly sample a chunk of chunk_size from the file
            chunk_size = self.chunk_size + 1
            if self.sample_rate is not None and sample_rate != self.sample_rate:
                base_chunk_size = chunk_size
                chunk_size = int(chunk_size * sample_rate / self.sample_rate)
            offset = torch.randint(0, max(1, file_length - chunk_size*3), (1,)).item()
            # print(f"Loading {path} at offset {offset} for chunk size {chunk_size}")
            wav, sr = torchaudio.load(path, normalize=True, frame_offset=offset, num_frames=chunk_size, backend="soundfile")
            if wav.shape[0] == 0 or wav.shape[1] == 0:
                # empty file, return first one
                # print(f"Empty file {path}, returning first sample instead.")
                return self[0]
            if self.sample_rate is not None and sample_rate != self.sample_rate:
                # print(f"Resampling from {sample_rate} to {self.sample_rate}")
                wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate).clamp(-1.0, 1.0)
                assert wav.shape[1] == base_chunk_size, f"Resampled wav length {wav.shape[1]} does not match expected chunk size {base_chunk_size}"
                chunk_size = base_chunk_size
            wav = quantize_unsigned_pcm_torch(wav, n_bits=self.max_bit_depth if self.max_bit_depth is not None else 24, kind=self.encoding)

            # randomly sample left or right channel
            if self.stereo_interleave:
                if wav.shape[0] < 2:
                    # if mono, duplicate channel
                    wav = torch.cat([wav, wav], dim=0)
                # put left, then right or right then left
                interleaved = torch.zeros(wav.shape[1] * 2, dtype=wav.dtype)
                if torch.rand(1).item() < 0.5:
                    interleaved[:wav.shape[1]] = wav[0]
                    interleaved[wav.shape[1]:] = wav[1]
                else:
                    interleaved[:wav.shape[1]] = wav[1]
                    interleaved[wav.shape[1]:] = wav[0]
                wav = interleaved
            else:
                # if stereo, randomly pick one channel
                if wav.shape[0] == 2:
                    if torch.rand(1).item() < 0.5:
                        wav = wav[1]  # take right channel only
                    else:
                        wav = wav[0]  # take left channel only
                else:
                    wav = wav[0]  # mono
            # split bits into 3 bytes if 24 bits, or 2 bytes if 16 bits, or 1 byte if 8 bits
            if self.max_bit_depth == 24 or self.max_bit_depth is None:
                bit1 = msb_torch(wav, orig_n_bits=24, n_bits=self.msb_n_bits)
                if self.msb_n_bits < 24:
                    bit2 = (wav >> 8) & 0xFF
                    bit3 = lsb_torch(wav, n_bits=24 - self.msb_n_bits - 8)
                    # if the original bit depth is 16 or 8 (or capped lower), mask the LSBs
                    effective_bits = metadata['bits_per_sample']
                    if self.bit_depth_cap is not None:
                        effective_bits = min(effective_bits, self.bit_depth_cap)
                    if effective_bits <= 16:
                        # drop bit3: no data below 16-bit precision
                        bit3 = torch.full_like(bit3, self.lb_mask_token)
                    if effective_bits <= 8:
                        # drop bit2 as well: no data below 8-bit precision
                        bit2 = torch.full_like(bit2, self.lb_mask_token)
                    # drop lower bytes with probability lb_dropout
                    if isinstance(self.lb_dropout, list):
                        if len(self.lb_dropout) != 2:
                            raise ValueError("lb_dropout list must have length 2")
                        if torch.rand(1).item() < self.lb_dropout[0]:
                            bit2 = torch.full_like(bit2, self.lb_mask_token)
                            bit3 = torch.full_like(bit3, self.lb_mask_token)
                        elif torch.rand(1).item() < self.lb_dropout[1]:
                            bit3 = torch.full_like(bit3, self.lb_mask_token)
                    else:
                        if torch.rand(1).item() < self.lb_dropout:
                            bit2 = torch.full_like(bit2, self.lb_mask_token)
                            bit3 = torch.full_like(bit3, self.lb_mask_token)
                        elif torch.rand(1).item() < self.lb_dropout:
                            bit3 = torch.full_like(bit3, self.lb_mask_token)
                    wav = torch.stack([bit1, bit2, bit3], dim=1).view(-1)
                else:
                    wav = bit1
            elif self.max_bit_depth == 16:

                bit1 = msb_torch(wav, orig_n_bits=16, n_bits=self.msb_n_bits)
                if self.msb_n_bits < 16:
                    bit2 = lsb_torch(wav, n_bits=16 - self.msb_n_bits)
                    # if the original bit depth is 8 (or capped to 8), mask the LSB
                    effective_bits = metadata['bits_per_sample']
                    if self.bit_depth_cap is not None:
                        effective_bits = min(effective_bits, self.bit_depth_cap)
                    if effective_bits <= 8:
                        bit2 = torch.full_like(bit2, self.lb_mask_token)
                    # drop lower byte with probability lb_dropout
                    if torch.rand(1).item() < self.lb_dropout:
                        bit2 = torch.full_like(bit2, self.lb_mask_token)
                    wav = torch.stack([bit1, bit2], dim=1).view(-1)
                else:
                    wav = bit1
            elif self.max_bit_depth == 8:
                bit1 = lsb_torch(wav, n_bits=8)
                wav = bit1

            # if sr != self.sample_rate:
            #     print(f"Resampling from {sr} to {self.sample_rate}")
            #     wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            if len(wav) < self.chunk_size+1:
                print(f"Padding audio from {len(wav)} to {self.chunk_size+1}")
                wav = torch.nn.functional.pad(wav, (0, self.chunk_size+1 - len(wav)), mode='constant', value=q_zero(bits=8))

            seq_len = self.chunk_size * (self.max_bit_depth // 8 if self.max_bit_depth is not None else 3) * (2 if self.stereo_interleave else 1)
            seq_len = min(seq_len + 1, len(wav))
            # if less than seq_len, pad with mask token
            if len(wav) < seq_len:
                wav = torch.nn.functional.pad(wav, (0, 3073 - len(wav)), mode='constant', value=self.lb_mask_token)
                assert len(wav) == 3073, f"After padding, wav length {wav.shape} does not match expected seq_len {seq_len}"
            # .clone() ensures the returned tensor owns its storage independently.
            # Without it, wav[:seq_len] is a view; PyTorch's shared-memory collation
            # in multi-worker DataLoaders calls resize_() on the underlying storage,
            # which raises "Trying to resize storage that is not resizable".
            tokens = wav[:3073].clone()
            return tokens
        except Exception as e:
            print(f"Error loading {self.files[idx][0]}: {e}. Returning a random other sample.")
            return self[0]


def custom_collate_fn(batch):
    # print([x.size() for x in batch])
    # batch is a list of tensors of shape (seq_len,)
    # stack into a tensor of shape (batch_size, seq_len)
    return torch.stack([x.contiguous().clone() for x in batch if x.shape[0] == 3073], dim=0)

# =====================
# Multi-Dataset (Trilobyte unified, all bit-depths)
# =====================
class MultiTriloByteDataset(Dataset):
    """
    Combines multiple TriloByteDataset instances into a single dataset, normalizing
    all audio to 24-bit Trilobyte representation regardless of native bit depth.

    For 16-bit files: least significant byte (bit3) is set to the null/mask token (768).
    For 8-bit files:  two least significant bytes (bit2, bit3) are set to the mask token.

    All sub-datasets produce sequences of exactly the same token length:
        global_chunk_size * 6 + 1 tokens  (the +1 is the LM causal shift window)

    Stereo dataset config: loads global_chunk_size audio samples per channel
    Mono dataset config:   loads global_chunk_size * 2 audio samples
                           (same token count since 2x samples × 3 bytes = 6x)

    Each entry in dataset_configs is a dict with keys:
        data_dir            (str, required)
        metadata_path       (str, required)
        sample_rate         (int, optional – None means use file native rate)
        encoding            (str, optional – 'linear' or 'mu-law', default 'linear')
        stereo_interleave   (bool, optional, default False)
    """

    def __init__(
        self,
        dataset_configs: List[dict],
        global_chunk_size: int,
        epoch_expansion_factor: int = 1,
        lb_dropout: Union[float, List[float]] = 0.0,
    ):
        self.global_chunk_size = global_chunk_size
        self.sub_datasets = []

        for cfg in dataset_configs:
            stereo = cfg.get('stereo_interleave', False)
            # Stereo: chunk_size * 3bytes * 2ch = chunk_size*6 tokens
            # Mono:   (chunk_size*2) * 3bytes * 1ch = chunk_size*6 tokens
            inner_chunk_size = global_chunk_size if stereo else global_chunk_size * 2
            # per-dataset epoch_expansion_factor overrides the global default
            ds_epoch_expansion = cfg.get('epoch_expansion_factor', epoch_expansion_factor)

            ds = TriloByteDataset(
                data_dir=cfg['data_dir'],
                chunk_size=inner_chunk_size,
                sample_rate=cfg.get('sample_rate', None),
                epoch_expansion_factor=ds_epoch_expansion,
                stereo_interleave=stereo,
                lb_dropout=lb_dropout,
                # max_bit_depth=None: forces 24-bit Trilobyte processing, skips
                # the bit-depth file filter, and uses mask token 768 for LSBs
                # that don't exist in the original lower-bit-depth signal.
                max_bit_depth=None,
                metadata_path=cfg['metadata_path'],
                encoding=cfg.get('encoding', 'linear'),
                bit_depth_cap=cfg.get('bit_depth_cap', None),
            )
            self.sub_datasets.append(ds)

        self._concat = torch.utils.data.ConcatDataset(self.sub_datasets)

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]


# =====================
# Lightning Module
# =====================
class GPTAudioLightningModule(pl.LightningModule):
    def __init__(self, model_name='gpt2', use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1, lora_target_modules='c_attn', lr=3e-4, weight_decay=0.1, warmup_steps=1000, max_steps=-1, min_lr=1e-5, log_last_p=0.5, max_bit_depth=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if type(self.hparams.max_bit_depth) is str:
            max_bit_depth = int(self.hparams.max_bit_depth.split('_')[0])
            self.hparams.msb_n_bits = int(self.hparams.max_bit_depth.split('_')[1])
            self.hparams.max_bit_depth = max_bit_depth
        else:
            self.hparams.msb_n_bits = 8
            max_bit_depth = self.hparams.max_bit_depth

        config = GPT2Config.from_pretrained(model_name)
        # if split_bit == 2:
        #     config.vocab_size = 512  # 256 high bits + 256 low bits
        # elif split_bit == 4:
        #     config.vocab_size = 64  # 16 + 16 + 16 + 16
        # elif split_bit == 3:
        #     config.vocab_size = 288  # 256 + 16 + 16
        # elif only_lower_bits:
        #     config.vocab_size = 256  # only low 8 bits
        # else:
        #     config.vocab_size = 65536  # 16-bit tokens
        match max_bit_depth:
            case 24 | None:
                config.vocab_size = 768
                print(f"[VOCAB SIZE] Using vocab size {config.vocab_size} for max bit depth {max_bit_depth} with MSB bits {self.hparams.msb_n_bits} and LSB bits {24 - self.hparams.msb_n_bits}")
            case 16:
                config.vocab_size = msb_bits_2_vocab_size(self.hparams.msb_n_bits)
                print(f"[VOCAB SIZE] Using vocab size {config.vocab_size} for max bit depth {max_bit_depth} with MSB bits {self.hparams.msb_n_bits} and LSB bits {16 - self.hparams.msb_n_bits}")
            case 8:
                config.vocab_size = 256  # 256

        if kwargs.get("lb_dropout", 0.0) > 0.0 or max_bit_depth is None:
            config.vocab_size += 1  # add mask token for lower bits dropout
            self.lb_mask_token = config.vocab_size - 1
        config.max_position_embeddings = 6145
        self.model = GPT2LMHeadModel(config)

        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[lora_target_modules],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.log_last_p = log_last_p
        self.validation_outputs = []
        self.val_log_ctr = 0

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        input_tokens = batch
        outputs = self(input_tokens, labels=input_tokens)
        loss = outputs.loss
        logits = outputs.logits.detach()
        bpb = loss / np.log(2)

        # Compute loss per index
        with torch.no_grad():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tokens[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.shape)

            avg_loss_per_pos = token_losses.mean(0)
            avg_bpb_per_pos = avg_loss_per_pos / np.log(2)
            last_p_idx = int(len(avg_loss_per_pos) * (1 - self.log_last_p))
            last_p_loss = avg_loss_per_pos[last_p_idx:].mean()
            last_p_bpb = avg_bpb_per_pos[last_p_idx:].mean()

        self.log('train/loss', loss,  on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/bpb', bpb,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/last_p_bpb', last_p_bpb, on_step=True, on_epoch=True, prog_bar=True)

        # occasionally log the whole per-index arrays to WandB
        step = int(self.global_step if hasattr(self, 'global_step') else 0)
        if step % 500 == 0:
            self._maybe_log_arrays_to_wandb('train', avg_loss_per_pos.detach().cpu().numpy(), avg_bpb_per_pos.detach().cpu().numpy(), step)

        return loss
    

    def validation_step(self, batch, batch_idx):
        input_tokens = batch
        outputs = self(input_tokens, labels=input_tokens)
        loss = outputs.loss
        logits = outputs.logits.detach()
        bpb = loss / np.log(2)

        # Compute loss per index
        with torch.no_grad():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tokens[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.shape)

            avg_loss_per_pos = token_losses.mean(0)
            avg_bpb_per_pos = avg_loss_per_pos / np.log(2)
            last_p_idx = int(len(avg_loss_per_pos) * (1 - self.log_last_p))
            last_p_loss = avg_loss_per_pos[last_p_idx:].mean()
            last_p_bpb = avg_bpb_per_pos[last_p_idx:].mean()

            out_d = {
                'per_index_loss': avg_loss_per_pos.detach().cpu(),
                'per_index_bpb': avg_bpb_per_pos.detach().cpu(),
            }

            if self.hparams.lb_dropout > 0.0 and (self.hparams.max_bit_depth is None or self.hparams.max_bit_depth > 8):
                # replace lower bit with mask token for input and target, run it on the model, and then get the loss for most significant bits only
                mask_token = self.lb_mask_token
                masked_input = input_tokens.clone()
                if self.hparams.max_bit_depth == 24 or self.hparams.max_bit_depth is None:
                    # we're gonna do two passes here, one for dropping byte 2 and byte 3 (i.e. 8 bit), and one for dropping byte 3 only (i.e. 16 bit)
                    masked_input_8b = masked_input.clone()
                    masked_input_16b = masked_input.clone()
                    # replace every second and third token (the lower bits) with the mask token
                    masked_input_8b[..., 1::3] = mask_token
                    masked_input_8b[..., 2::3] = mask_token
                    # for 16 bit dropout, only replace every third token
                    masked_input_16b[..., 2::3] = mask_token
                    masked_outputs_8b = self(masked_input_8b, labels=masked_input_8b)
                    shift_masked_logits_8b = masked_outputs_8b.logits[..., :-1, :].contiguous()
                    shift_masked_labels_8b = masked_input_8b[..., 1:].contiguous()
                    masked_token_losses_8b = loss_fct(shift_masked_logits_8b.view(-1, shift_masked_logits_8b.size(-1)), shift_masked_labels_8b.view(-1))
                    masked_token_losses_8b = masked_token_losses_8b.view(shift_masked_labels_8b.shape)
                    msb_token_losses_8b = masked_token_losses_8b[:, 2::3]  # take only the losses for the most significant bits
                    masked_avg_loss_8b = msb_token_losses_8b.mean()
                    masked_avg_bpb_8b = masked_avg_loss_8b / np.log(2)
                    out_d['msb_loss_8b'] = masked_avg_loss_8b.detach().cpu()
                    out_d['msb_bpb_8b'] = masked_avg_bpb_8b.detach().cpu()

                    masked_outputs_16b = self(masked_input_16b, labels=masked_input_16b)
                    shift_masked_logits_16b = masked_outputs_16b.logits[..., :-1, :].contiguous()
                    shift_masked_labels_16b = masked_input_16b[..., 1:].contiguous()
                    masked_token_losses_16b = loss_fct(shift_masked_logits_16b.view(-1, shift_masked_logits_16b.size(-1)), shift_masked_labels_16b.view(-1))
                    masked_token_losses_16b = masked_token_losses_16b.view(shift_masked_labels_16b.shape)
                    # take losses for byte1 and byte2 (i.e. most significant 16 bits)
                    msb_token_losses_16b = torch.cat([masked_token_losses_16b[:, 0::3], masked_token_losses_16b[:, 2::3]], dim=1)
                    masked_avg_loss_16b = msb_token_losses_16b.mean()
                    masked_avg_bpb_16b = masked_avg_loss_16b / np.log(2)
                    out_d['msb_loss_16b'] = masked_avg_loss_16b.detach().cpu()
                    out_d['msb_bpb_16b'] = masked_avg_bpb_16b.detach().cpu()


                # data is represented as 2 bytes per sample
                elif self.hparams.max_bit_depth == 16:
                    masked_input_8b = masked_input.clone()
                    masked_input_8b[..., 1::2] = mask_token
                    
                    masked_outputs_8b = self(masked_input_8b, labels=masked_input_8b)
                    shift_masked_logits_8b = masked_outputs_8b.logits[..., :-1, :].contiguous()
                    shift_masked_labels_8b = masked_input_8b[..., 1:].contiguous()
                    masked_token_losses_8b = loss_fct(shift_masked_logits_8b.view(-1, shift_masked_logits_8b.size(-1)), shift_masked_labels_8b.view(-1))
                    masked_token_losses_8b = masked_token_losses_8b.view(shift_masked_labels_8b.shape)
                    msb_token_losses_8b = masked_token_losses_8b[:, 1::2]
                    masked_avg_loss_8b = msb_token_losses_8b.mean()
                    masked_avg_bpb_8b = masked_avg_loss_8b / np.log(2)
                    out_d['msb_loss_8b'] = masked_avg_loss_8b.detach().cpu()
                    out_d['msb_bpb_8b'] = masked_avg_bpb_8b.detach().cpu()

                    

        self.validation_outputs.append(out_d)

        return {
            'val_loss': loss.detach().cpu(),
            'val_bpb': bpb.detach().cpu(),
            'val_last_p_bpb': last_p_bpb.detach().cpu(),
            'val_per_index_loss': avg_loss_per_pos.detach().cpu(),
            'val_per_index_bpb': avg_bpb_per_pos.detach().cpu(),
        }

    def on_validation_epoch_end(self):

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
        start_idx = int(L * (1.0 - float(self.log_last_p)))
        last_p_bpb = float(mean_per_index_bpb[start_idx:].mean())
        self.log('val/last_p_bpb', last_p_bpb, on_epoch=True, prog_bar=True)
        

        # log loss/bit-per-byte
        mean_loss = float(mean_per_index_loss.mean())
        self.log('val/loss', mean_loss, on_epoch=True, prog_bar=True)

        if self.hparams.get("lb_dropout", 0.0) > 0.0 and (self.hparams.max_bit_depth is None or self.hparams.max_bit_depth > 8):
            if self.hparams.max_bit_depth == 24 or self.hparams.max_bit_depth is None:
                # log the msb-only loss/bpb for 8 bit dropout
                msb_losses_8b = []
                msb_bpbs_8b = []
                msb_losses_16b = []
                msb_bpbs_16b = []
                for o in self.validation_outputs:
                    if 'msb_loss_8b' in o and 'msb_bpb_8b' in o:
                        msb_losses_8b.append(o['msb_loss_8b'].numpy())
                        msb_bpbs_8b.append(o['msb_bpb_8b'].numpy())
                    if 'msb_loss_16b' in o and 'msb_bpb_16b' in o:
                        msb_losses_16b.append(o['msb_loss_16b'].numpy())
                        msb_bpbs_16b.append(o['msb_bpb_16b'].numpy())
                if len(msb_losses_8b) > 0:
                    mean_msb_loss_8b = np.stack(msb_losses_8b, axis=0).mean()
                    mean_msb_bpb_8b = np.stack(msb_bpbs_8b, axis=0).mean()
                    self.log('val/msb_loss_8b', float(mean_msb_loss_8b), on_epoch=True, prog_bar=True)
                    self.log('val/msb_bpb_8b', float(mean_msb_bpb_8b), on_epoch=True, prog_bar=True)
                if len(msb_losses_16b) > 0:
                    mean_msb_loss_16b = np.stack(msb_losses_16b, axis=0).mean()
                    mean_msb_bpb_16b = np.stack(msb_bpbs_16b, axis=0).mean()
                    self.log('val/msb_loss_16b', float(mean_msb_loss_16b), on_epoch=True, prog_bar=True)
                    self.log('val/msb_bpb_16b', float(mean_msb_bpb_16b), on_epoch=True, prog_bar=True)
            elif self.hparams.max_bit_depth == 16:
                # log the msb-only loss/bpb for 8 bit dropout
                msb_losses_8b = []
                msb_bpbs_8b = []
                for o in self.validation_outputs:
                    if 'msb_loss_8b' in o and 'msb_bpb_8b' in o:
                        msb_losses_8b.append(o['msb_loss_8b'].numpy())
                        msb_bpbs_8b.append(o['msb_bpb_8b'].numpy())
                if len(msb_losses_8b) > 0:
                    mean_msb_loss_8b = np.stack(msb_losses_8b, axis=0).mean()
                    mean_msb_bpb_8b = np.stack(msb_bpbs_8b, axis=0).mean()
                    self.log('val/msb_loss_8b', float(mean_msb_loss_8b), on_epoch=True, prog_bar=True)
                    self.log('val/msb_bpb_8b', float(mean_msb_bpb_8b), on_epoch=True, prog_bar=True)

        # lo losses for msb and lsb separately
        if self.hparams.max_bit_depth == 16:
            # 2 bytes per sample
            mean_msb_loss = float(mean_per_index_loss[1::2].mean())
            mean_msb_bpb = float(mean_per_index_bpb[1::2].mean())
            mean_lsb_loss = float(mean_per_index_loss[::2].mean())
            mean_lsb_bpb = float(mean_per_index_bpb[::2].mean())
            self.log('val/msb_loss', float(mean_msb_loss), on_epoch=True, prog_bar=True)
            self.log('val/msb_bpb', float(mean_msb_bpb), on_epoch=True, prog_bar=True)
            self.log('val/lsb_loss', float(mean_lsb_loss), on_epoch=True, prog_bar=True)
            self.log('val/lsb_bpb', float(mean_lsb_bpb), on_epoch=True, prog_bar=True)
            # log number of msb/lsb bits / mean bpb as bar plot
            self.log('val/msb_compression_rate', float(self.hparams.msb_n_bits / mean_msb_bpb), on_epoch=True, prog_bar=True)
            self.log('val/lsb_compression_rate', float((16 - self.hparams.msb_n_bits) / mean_lsb_bpb), on_epoch=True, prog_bar=True)
            # also log the msb-only loss/bpb
            # msb_losses = []
            # msb_bpbs = []
            # for o in self.validation_outputs:
            #     if 'msb_loss' in o and 'msb_bpb' in o:
            #         msb_losses.append(o['msb_loss'].numpy())
            #         msb_bpbs.append(o['msb_bpb'].numpy())
            # if len(msb_losses) > 0:
            #     mean_msb_loss = np.stack(msb_losses, axis=0).mean()
            #     mean_msb_bpb = np.stack(msb_bpbs, axis=0).mean()
            #     self.log('val/msb_loss', float(mean_msb_loss), on_epoch=True, prog_bar=True)
            #     self.log('val/msb_bpb', float(mean_msb_bpb), on_epoch=True, prog_bar=True)
        self.validation_outputs = []  # reset for next epoch
    

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
                # if not self.hparams.split_bit:
                #     fig = Figure(figsize=(4.145, 8.29), dpi=100, tight_layout=True)
                #     canvas = FigureCanvasAgg(fig)
                #     ax = fig.add_subplot(2, 1, 1)
                #     ax.plot(per_index_loss_arr, label='Per-index Loss', color='C0')
                #     ax.set_title(f'{stage} Per-index Loss')
                #     ax.set_xlabel('Index (sample)')
                #     ax.set_ylabel('Loss (nats)')
                #     ax.grid(True)
                #     ax = fig.add_subplot(2, 1, 2)
                #     ax.plot(per_index_bpb_arr, label='Per-index Bits-per-byte', color='C1')
                #     ax.set_title(f'{stage} Per-index Bits-per-byte')
                #     ax.set_xlabel('Index (sample)')
                #     ax.set_ylabel('Bits-per-byte')
                #     ax.grid(True)
                #     plt.tight_layout()
                #     canvas.draw()
                #     rgba = np.asarray(canvas.buffer_rgba())
                #     im = Image.fromarray(rgba)
                #     exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                # else:
                    # plot 4 subplots: loss high bits, loss low bits, bpb high bits, bpb low bits
                match self.hparams.max_bit_depth:
                    case 24 | None:
                        # plot for 24bit (3 bytes), 16bit (2 bytes), and 8bit (1 byte)
                        # plot bpb for each byte in a grid
                        fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(3, 1, 1)
                        ax.plot(per_index_bpb_arr[2::3], label='Per-index Bits-per-byte Byte 1 (MSB)', color='C0')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Byte 1 (MSB)')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(3, 1, 2)
                        ax.plot(per_index_bpb_arr[::3], label='Per-index Bits-per-byte Byte 2', color='C1')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Byte 2')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(3, 1, 3)
                        ax.plot(per_index_bpb_arr[1::3], label='Per-index Bits-per-byte Byte 3 (LSB)', color='C2')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Byte 3 (LSB)')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        plt.tight_layout()
                        canvas.draw()
                        rgba = np.asarray(canvas.buffer_rgba())
                        im = Image.fromarray(rgba)
                        exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    case 16:
                        # plot bpb for each byte in a grid, only 16bit and 8bit
                        fig = Figure(figsize=(6, 8.29), dpi=100, tight_layout=True)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(2, 1, 1)
                        ax.plot(per_index_bpb_arr[1::2], label='Per-index Bits-per-byte Byte 1 (MSB)', color='C0')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Byte 1 (MSB)')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 1, 2)
                        ax.plot(per_index_bpb_arr[::2], label='Per-index Bits-per-byte Byte 2 (LSB)', color='C1')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Byte 2 (LSB)')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        plt.tight_layout()
                        canvas.draw()
                        rgba = np.asarray(canvas.buffer_rgba())
                        im = Image.fromarray(rgba)
                        exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    case 8:
                        pass
                    # if self.hparams.split_bit == 2 or self.hparams.split_bit == True:
                    #     fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                    #     canvas = FigureCanvasAgg(fig)
                    #     ax = fig.add_subplot(2, 2, 1)
                    #     ax.plot(per_index_loss_arr[1::2], label='Per-index Loss High Bits', color='C0')
                    #     ax.set_title(f'{stage} Per-index Loss High Bits')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Loss (nats)')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 2)
                    #     ax.plot(per_index_loss_arr[::2], label='Per-index Loss Low Bits', color='C1')
                    #     ax.set_title(f'{stage} Per-index Loss Low Bits')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Loss (nats)')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 3)
                    #     ax.plot(per_index_bpb_arr[1::2], label='Per-index Bits-per-byte High Bits', color='C2')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte High Bits')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 4)
                    #     ax.plot(per_index_bpb_arr[::2], label='Per-index Bits-per-byte Low Bits', color='C3')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Low Bits')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     plt.tight_layout()
                    #     canvas.draw()
                    #     rgba = np.asarray(canvas.buffer_rgba())
                    #     im = Image.fromarray(rgba)
                    #     exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    # elif self.hparams.split_bit == 4:
                    #     # do 2x2 grid of only bpb plots, for each of the 4 bit positions
                    #     fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                    #     canvas = FigureCanvasAgg(fig)
                    #     ax = fig.add_subplot(2, 2, 1)
                    #     ax.plot(per_index_bpb_arr[::4], label='Per-index Bits-per-byte Bit 3', color='C0')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Bit 3')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 2)
                    #     ax.plot(per_index_bpb_arr[1::4], label='Per-index Bits-per-byte Bit 2', color='C1')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Bit 2')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 3)
                    #     ax.plot(per_index_bpb_arr[2::4], label='Per-index Bits-per-byte Bit 1', color='C2')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Bit 1')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 4)
                    #     ax.plot(per_index_bpb_arr[3::4], label='Per-index Bits-per-byte Bit 0', color='C3')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Bit 0')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     plt.tight_layout()
                    #     canvas.draw()
                    #     rgba = np.asarray(canvas.buffer_rgba())
                    #     im = Image.fromarray(rgba)
                    #     exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    # elif self.hparams.split_bit == 3:
                    #     # do first plot for 8-bit part, second plot for 4-bit part, third plot for last 4-bit part, and then 4th plot for all combined
                    #     fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                    #     canvas = FigureCanvasAgg(fig)
                    #     ax = fig.add_subplot(2, 2, 1)
                    #     ax.plot(per_index_bpb_arr[::3], label='Per-index Bits-per-byte Big 8-bit Part', color='C0')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Big 8-bit Part')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 2)
                    #     ax.plot(per_index_bpb_arr[1::3], label='Per-index Bits-per-byte Middle 4-bit Part', color='C1')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Middle 4-bit Part')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 3)
                    #     ax.plot(per_index_bpb_arr[2::3], label='Per-index Bits-per-byte Smallest 4-bit Part', color='C2')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte Smallest 4-bit Part')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     ax = fig.add_subplot(2, 2, 4)
                    #     ax.plot(per_index_bpb_arr, label='Per-index Bits-per-byte All Combined', color='C3')
                    #     ax.set_title(f'{stage} Per-index Bits-per-byte All Combined')
                    #     ax.set_xlabel('Index (sample)')
                    #     ax.set_ylabel('Bits-per-byte')
                    #     ax.grid(True)
                    #     plt.tight_layout()
                    #     canvas.draw()
                    #     rgba = np.asarray(canvas.buffer_rgba())
                    #     im = Image.fromarray(rgba)
                    #     exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)


            else:
                # Not a WandB logger: as a fallback, log first few per-index values to Lightning logs
                for i in range(min(10, len(per_index_loss_arr))):
                    self.log(f'{stage}/per_index_loss_idx_{i}', float(per_index_loss_arr[i]), on_epoch=True)
        except Exception as ex:
            print("_maybe_log_arrays_to_wandb failed:", ex)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = (
            self.hparams.max_steps if self.hparams.max_steps > 0 else self.trainer.estimated_stepping_batches
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# =====================
# Data Module
# =====================
class MonoDataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir, val_data_dir='',batch_size=4, num_workers=4, chunk_size=1024, sample_rate=44100, split_bit=False, only_lower_bits=False, train_p=1.0, stereo_interleave=False, lb_dropout=0.0, epoch_expansion_factor=10, max_bit_depth=None, train_metadata_path=None, val_metadata_path=None, encoding='linear'):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.split_bit = split_bit
        self.only_lower_bits = only_lower_bits
        self.train_p = train_p
        self.stereo_interleave = stereo_interleave
        self.lb_dropout = lb_dropout
        self.epoch_expansion_factor = epoch_expansion_factor
        self.max_bit_depth = max_bit_depth
        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.encoding = encoding

    def setup(self, stage=None):
        train_dir = self.train_data_dir
        valid_dir = self.val_data_dir

        # If explicit train/valid subdirectories exist, use them
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            print("HEY, WE'RE USING EXPLICIT TRAIN/VALID DIRECTORIES")
            # train_full = MonoWavChunkDataset(train_dir, chunk_size=self.chunk_size, bit_split=self.split_bit, only_lower_bits=self.only_lower_bits, stereo_interleave=self.stereo_interleave, lb_dropout=self.lb_dropout, epoch_expansion_factor=self.epoch_expansion_factor)
            # val_ds = MonoWavChunkDataset(valid_dir, chunk_size=self.chunk_size, bit_split=self.split_bit, only_lower_bits=self.only_lower_bits, stereo_interleave=self.stereo_interleave, lb_dropout=0.0, epoch_expansion_factor=self.epoch_expansion_factor)
            train_full = TriloByteDataset(train_dir, chunk_size=self.chunk_size, sample_rate=self.sample_rate, stereo_interleave=self.stereo_interleave, lb_dropout=self.lb_dropout, epoch_expansion_factor=self.epoch_expansion_factor, max_bit_depth=self.max_bit_depth, metadata_path=self.train_metadata_path, encoding=self.encoding)
            val_ds = TriloByteDataset(valid_dir, chunk_size=self.chunk_size, sample_rate=self.sample_rate, stereo_interleave=self.stereo_interleave, lb_dropout=0.0, epoch_expansion_factor=self.epoch_expansion_factor, max_bit_depth=self.max_bit_depth, metadata_path=self.val_metadata_path, encoding=self.encoding)

            if self.train_p < 1.0:
                n = len(train_full)
                keep = max(1, int(n * self.train_p))
                indices = list(range(n))
                random.shuffle(indices)
                indices = indices[:keep]
                train_ds = torch.utils.data.Subset(train_full, indices)
            else:
                train_ds = train_full

            self.train_ds = train_ds
            self.val_ds = val_ds
        else:
            print("HEY, WE'RE USING A SINGLE DIRECTORY FOR TRAIN/VALID, IF YOU DIDNT SET UP SPLITS YET, THIS WILL DO IT FOR YOU")
            # Fallback: use single directory and split internally (original behavior)
            dataset = TriloByteDataset(self.train_data_dir, chunk_size=self.chunk_size, sample_rate=self.sample_rate, stereo_interleave=self.stereo_interleave, lb_dropout=self.lb_dropout, epoch_expansion_factor=self.epoch_expansion_factor, max_bit_depth=self.max_bit_depth, metadata_path=self.train_metadata_path, encoding=self.encoding)
            n = len(dataset)
            frac_n = int(n * self.train_p)
            n_train = int(0.9 * frac_n)
            # manually split this dataset into train and validation sets (90% train, 10% val) by subsetting the dataset.files list
            train_files = dataset.files[:n_train]
            val_files = dataset.files[n_train:frac_n]
            train_dataset = TriloByteDataset(self.train_data_dir, chunk_size=self.chunk_size, sample_rate=self.sample_rate, stereo_interleave=self.stereo_interleave, lb_dropout=self.lb_dropout, epoch_expansion_factor=self.epoch_expansion_factor, max_bit_depth=self.max_bit_depth, metadata_path=self.train_metadata_path, encoding=self.encoding)
            val_dataset = TriloByteDataset(self.train_data_dir, chunk_size=self.chunk_size, sample_rate=self.sample_rate, stereo_interleave=self.stereo_interleave, lb_dropout=0.0, epoch_expansion_factor=self.epoch_expansion_factor, max_bit_depth=self.max_bit_depth, metadata_path=self.train_metadata_path, encoding=self.encoding)
            train_dataset.files = train_files
            val_dataset.files = [f for f in val_files if f[1]['bits_per_sample'] == int(self.max_bit_depth.split("_")[0])]  # filter val files to only include those with the correct bit depth (in case of mixed bit depths in the same directory)
            print(f"Total files: {len(dataset.files)}, Train files: {len(train_dataset.files)}, Val files: {len(val_dataset.files)}")
            self.train_ds = train_dataset
            self.val_ds = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# =====================
# Multi-Dataset Data Module
# =====================
class MultiDataModule(pl.LightningDataModule):
    """
    Data module that reads a JSON config describing multiple datasets and combines
    them into a single MultiTriloByteDataset for training and validation.

    The model is always run in 24-bit Trilobyte mode (max_bit_depth=None),
    so make sure to initialize GPTAudioLightningModule with max_bit_depth=None
    when using this data module.

    Config file format (JSON):
    {
        "datasets": [
            {
                "train_data_dir":     "/path/to/train",
                "val_data_dir":       "/path/to/val",
                "train_metadata_path": "/path/to/train_meta.json",
                "val_metadata_path":   "/path/to/val_meta.json",
                "sample_rate":        44100,       // optional, null uses file rate
                "encoding":           "linear",    // optional, default "linear"
                "stereo_interleave":  false        // optional, default false
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        dataset_config_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        global_chunk_size: int = 512,
        epoch_expansion_factor: int = 1,
        lb_dropout: Union[float, List[float]] = 0.0,
        train_p: float = 1.0,
    ):
        super().__init__()
        self.dataset_config_path = dataset_config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.global_chunk_size = global_chunk_size
        self.epoch_expansion_factor = epoch_expansion_factor
        self.lb_dropout = lb_dropout
        self.train_p = train_p

    def _build_dataset(self, data_dir, metadata_path, ds_cfg, lb_dropout):
        """Build a single TriloByteDataset with the shared normalization settings."""
        stereo = ds_cfg.get('stereo_interleave', False)
        inner_chunk_size = self.global_chunk_size if stereo else self.global_chunk_size * 2
        ds_epoch_expansion = ds_cfg.get('epoch_expansion_factor', self.epoch_expansion_factor)
        return TriloByteDataset(
            data_dir=data_dir,
            chunk_size=inner_chunk_size,
            sample_rate=ds_cfg.get('sample_rate', None),
            epoch_expansion_factor=ds_epoch_expansion,
            stereo_interleave=stereo,
            lb_dropout=lb_dropout,
            max_bit_depth=None,
            metadata_path=metadata_path,
            encoding=ds_cfg.get('encoding', 'linear'),
            bit_depth_cap=ds_cfg.get('bit_depth_cap', None),
        )

    def setup(self, stage=None):
        config = json.load(open(self.dataset_config_path))

        train_sub_datasets = []
        val_sub_datasets = []

        for key, ds_cfg in config['datasets'].items():
            print(f"[MultiDataModule] Setting up dataset '{key}' with config: {ds_cfg}")
            train_dir = ds_cfg['train_data_dir']
            val_dir = ds_cfg.get('val_data_dir', '')
            train_meta = ds_cfg['train_metadata_path']
            val_meta = ds_cfg.get('val_metadata_path', '')

            if os.path.isdir(train_dir) and os.path.isdir(val_dir):
                # Explicit separate train/val directories
                print(f"[MultiDataModule] Using explicit train/val dirs for {train_dir}")
                train_ds = self._build_dataset(train_dir, train_meta, ds_cfg, self.lb_dropout)
                val_ds = self._build_dataset(val_dir, val_meta, ds_cfg, 0.0)
            else:
                # No val dir: split training data 90/10 (mirroring MonoDataModule)
                print(f"[MultiDataModule] No val dir for {train_dir}, splitting 90/10")
                full_ds = self._build_dataset(train_dir, train_meta, ds_cfg, self.lb_dropout)
                n = len(full_ds)
                n_train = int(0.9 * n)
                train_ds = self._build_dataset(train_dir, train_meta, ds_cfg, self.lb_dropout)
                val_ds = self._build_dataset(train_dir, train_meta, ds_cfg, 0.0)
                train_ds.files = full_ds.files[:n_train]
                val_ds.files = full_ds.files[n_train:]
                print(f"[MultiDataModule] Train: {len(train_ds.files)} files, Val: {len(val_ds.files)} files")

            train_sub_datasets.append(train_ds)
            val_sub_datasets.append(val_ds)

        train_full = torch.utils.data.ConcatDataset(train_sub_datasets)
        val_concat = torch.utils.data.ConcatDataset(val_sub_datasets)

        if self.train_p < 1.0:
            n = len(train_full)
            keep = max(1, int(n * self.train_p))
            indices = list(range(n))
            random.shuffle(indices)
            train_ds = torch.utils.data.Subset(train_full, indices[:keep])
        else:
            train_ds = train_full

        self.train_ds = train_ds
        self.val_ds = val_concat

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)


# =====================
# Training Entry Point
# =====================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default=None, help='Path to multi-dataset JSON config (mutually exclusive with --train_data_dir). Forces 24-bit Trilobyte model.')
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--val_data_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--sample_rate', type=int, default=None)
    parser.add_argument('--encoding', type=str, default='linear', help='Encoding type for quantization (linear or mu-law)')
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=500_000)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default='c_attn')
    parser.add_argument('--log_last_p', type=float, default=0.5)
    parser.add_argument('--project', type=str, default='t5_lnac')
    parser.add_argument('--split_bit', type=int, default=0, help='If >0, split each 16-bit sample into N parts (2 or 4) and interleave them')
    parser.add_argument('--only_lower_bits', action='store_true', help='Whether to use only the lower 8 bits of 16-bit samples')
    parser.add_argument('--train_p', type=float, default=1.0, help='Proportion of training data to use')
    parser.add_argument('--stereo_interleave', action='store_true', help='Whether to interleave stereo channels')
    parser.add_argument('--lb_dropout', type=float, default=0.0, help='Probability of dropping out lower bits when using bit-splitting')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a checkpoint to resume from')
    parser.add_argument('--ckpt_not_dropout', action='store_true', help='Whether the loaded checkpoint was trained without dropout (if so, we need to modify the state dict so that the model.transformer.wte.weight and model.lm_head.weight are 1 larger to account for the added mask token)')
    parser.add_argument('--epoch_expansion_factor', type=int, default=1, help='Factor to expand dataset size per epoch')
    parser.add_argument('--max_bit_depth', type=str, default=None, help='Maximum bit depth of audio data (8, 16, or 24). If not set, infer from data.')
    parser.add_argument('--train_metadata_path', type=str, default=None, help='Path to metadata file for the training dataset (if any)')
    parser.add_argument('--val_metadata_path', type=str, default='', help='Path to metadata file for the validation dataset (if any)')
    args = parser.parse_args()

    if args.dataset_config is None and args.train_data_dir is None:
        parser.error("Either --dataset_config or --train_data_dir must be provided.")

    # When using multi-dataset config, force 24-bit Trilobyte model (max_bit_depth=None)
    if args.dataset_config is not None:
        args.max_bit_depth = None

    if args.max_bit_depth is not None and "_" not in str(args.max_bit_depth):
        args.max_bit_depth = int(args.max_bit_depth)

    # set seeds
    pl.seed_everything(args.seed)

    wandb_logger = WandbLogger(project=args.project)

    model = GPTAudioLightningModule(**vars(args))

    if args.dataset_config is not None:
        dm = MultiDataModule(
            dataset_config_path=args.dataset_config,
            batch_size=args.batch_size,
            global_chunk_size=args.chunk_size,
            epoch_expansion_factor=args.epoch_expansion_factor,
            lb_dropout=args.lb_dropout,
            train_p=args.train_p,
            num_workers=args.num_workers
        )
    else:
        dm = MonoDataModule(args.train_data_dir, args.val_data_dir, batch_size=args.batch_size, chunk_size=args.chunk_size, sample_rate=args.sample_rate, split_bit=args.split_bit, only_lower_bits=args.only_lower_bits, train_p=args.train_p, stereo_interleave=args.stereo_interleave, lb_dropout=args.lb_dropout, epoch_expansion_factor=args.epoch_expansion_factor, max_bit_depth=args.max_bit_depth, train_metadata_path=args.train_metadata_path, val_metadata_path=args.val_metadata_path, encoding=args.encoding)


    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",          # what metric to monitor
        mode="min",                  # lower is better
        save_top_k=1,                # keep best 3 checkpoints
        save_last=True,              # always save the last epoch
        every_n_epochs=10,
        filename="gpt2audio-{epoch:02d}"
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        precision='bf16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )


    if args.ckpt_path is not None and args.ckpt_not_dropout:
        # we need to modify the checkpoint state dict to add the mask token weights (initialized to 0) to the wte and lm_head weights
        print("Modifying checkpoint state dict to add mask token weights...")
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        # find the size of the wte weight matrix and add a new row for the mask token
        wte_weight = state_dict['model.transformer.wte.weight']
        vocab_size, hidden_size = wte_weight.shape
        new_wte_weight = torch.zeros((vocab_size + 1, hidden_size))
        new_wte_weight[:-1, :] = wte_weight
        state_dict['model.transformer.wte.weight'] = new_wte_weight
        # do the same for the lm_head weight matrix
        lm_head_weight = state_dict['model.lm_head.weight']
        new_lm_head_weight = torch.zeros((vocab_size + 1, hidden_size))
        new_lm_head_weight[:-1, :] = lm_head_weight
        state_dict['model.lm_head.weight'] = new_lm_head_weight
        # save the modified checkpoint to a new file
        modified_ckpt_path = args.ckpt_path.replace('.ckpt', '_modified.ckpt')
        torch.save(ckpt, modified_ckpt_path)
        print(f"Modified checkpoint saved to {modified_ckpt_path}")
        args.ckpt_path = modified_ckpt_path
        # update the optimizer state if it exists, to account for the new parameters
        # reset the optimizer state since the parameter shapes have changed
        if 'optimizer_states' in ckpt:
            print("Resetting optimizer state due to changed parameter shapes...")
            ckpt['optimizer_states'] = []
            torch.save(ckpt, modified_ckpt_path)
            print(f"Optimizer state reset in modified checkpoint {modified_ckpt_path}")

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
