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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

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


# =====================
# Dataset for Mono Audio
# =====================
class MonoWavChunkDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, sample_rate=44100, bit_split=False, epoch_expansion_factor=10, only_lower_bits=False, stereo_interleave=False, lb_dropout=0.0):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.bit_split = bit_split
        self.epoch_expansion_factor = epoch_expansion_factor
        self.only_lower_bits = only_lower_bits
        self.stereo_interleave = stereo_interleave
        self.lb_dropout = lb_dropout

        # define mask token for lower bits dropout
        match self.bit_split:
            case 2:
                self.lb_mask_token = 512
            case 4:
                self.lb_mask_token = 64
            case 3:
                self.lb_mask_token = 272

        if len(self.files) == 0:
            raise ValueError("files is empty")
        print(f"MonoWavChunkDataset: {len(self.files)} files, chunk_size={self.chunk_size}")
        pth = 'musdbstereo_lengths_train.json' if 'train' in data_dir else 'musdbstereo_lengths_valid.json' if 'valid' in data_dir else 'musdbstereo_lengths.json'
        lengths = json.load(open(pth, 'r'))
        for ix, f in enumerate(tqdm(self.files)):
            self.files[ix] = (f, lengths[os.path.basename(f)])  # (path, num_samples)
        self.files = self.files * self.epoch_expansion_factor
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, file_length = self.files[idx]
        # randomly sample a chunk of chunk_size from the file
        chunk_size = self.chunk_size + 1
        offset = torch.randint(0, max(1, file_length - chunk_size), (1,)).item()
        wav, sr = torchaudio.load(path, normalize=False, frame_offset=offset, num_frames=chunk_size, backend="soundfile")
        # wav, sr = torchaudio.load(path, normalize=False)
        if wav.dtype != torch.int16:
            wav = linear_encode(wav, bits=16)
        else:
            wav = wav.long() + 32768  # 

        # randomly sample left or right channel
        if self.stereo_interleave:
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
        # if bit_split is set, split each 16-bit value into two 8-bit values representing the high and low bytes
        if self.bit_split:
            splits = self.bit_split if type(self.bit_split) is int else 2
            if splits == 2:
                high_bits = (wav >> 8) & 0xFF
                low_bits = wav & 0xFF
                # add 2^8 to the low bits to distinguish them from high bits
                low_bits += 256
                # interleave high and low bits
                if torch.rand(1).item() < self.lb_dropout:
                    low_bits = torch.full_like(low_bits, self.lb_mask_token)

                wav = torch.stack([high_bits, low_bits], dim=1).view(-1)
                assert torch.all(wav[0] == high_bits[0])
                assert torch.all(wav[1] == low_bits[0])
            elif splits == 4:
                byte3 = (wav >> 12) & 0x0F
                byte2 = (wav >> 8) & 0x0F
                byte1 = (wav >> 4) & 0x0F
                byte0 = wav & 0x0F
                # add 2^4, 2^8, 2^12 to distinguish them
                byte2 += 16
                byte1 += 32
                byte0 += 48
                wav = torch.stack([byte3, byte2, byte1, byte0], dim=1).view(-1)
                assert torch.all(wav[0] == byte3[0])
                assert torch.all(wav[1] == byte2[0])
                assert torch.all(wav[2] == byte1[0])
                assert torch.all(wav[3] == byte0[0])
            elif splits == 3:
                # first highest 8 bits, then next 4 bits, then lowest 4 bits
                byte2 = (wav >> 8) & 0xFF
                byte1 = (wav >> 4) & 0x0F
                byte0 = wav & 0x0F
                byte1 += 256
                byte0 += 272
                wav = torch.stack([byte2, byte1, byte0], dim=1).view(-1)
                assert torch.all(wav[0] == byte2[0])
                assert torch.all(wav[1] == byte1[0])
                assert torch.all(wav[2] == byte0[0])
        elif self.only_lower_bits:
            wav = wav & 0xFF  # keep only the lower 8 bits


        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if len(wav) < self.chunk_size+1:
            wav = torch.nn.functional.pad(wav, (0, self.chunk_size+1 - len(wav)), mode='constant', value=q_zero(bits=16))
        chunk = wav
        tokens = chunk.long()
        seq_len = self.chunk_size
        if self.bit_split:
            seq_len *= self.bit_split if type(self.bit_split) is int else 2
        if self.stereo_interleave:
            seq_len *= 2
        seq_len = min(seq_len + 1, len(tokens))
        tokens = tokens[:seq_len]
        # input_tokens = tokens
        # target_tokens = tokens[1:]
        return tokens


# =====================
# Lightning Module
# =====================
class GPTAudioLightningModule(pl.LightningModule):
    def __init__(self, model_name='gpt2', use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1, lora_target_modules='c_attn', lr=3e-4, weight_decay=0.1, warmup_steps=1000, max_steps=-1, min_lr=1e-5, log_last_p=0.5, split_bit=False, only_lower_bits=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = GPT2Config.from_pretrained(model_name)
        if split_bit == 2:
            config.vocab_size = 512  # 256 high bits + 256 low bits
        elif split_bit == 4:
            config.vocab_size = 64  # 16 + 16 + 16 + 16
        elif split_bit == 3:
            config.vocab_size = 288  # 256 + 16 + 16
        elif only_lower_bits:
            config.vocab_size = 256  # only low 8 bits
        else:
            config.vocab_size = 65536  # 16-bit tokens

        if kwargs.get("lb_dropout", 0.0) > 0.0 and split_bit:
            config.vocab_size += 1  # add mask token for lower bits dropout
            self.lb_mask_token = config.vocab_size - 1
        config.max_position_embeddings = 2049
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

            if self.hparams.lb_dropout > 0.0 and self.hparams.split_bit:
                # replace lower bit with mask token for input and target, run it on the model, and then get the loss for most significant bits only
                mask_token = self.lb_mask_token
                masked_input = input_tokens.clone()
                masked_target = input_tokens.clone()
                # replace every second token (the lower bits) with the mask token
                masked_input[..., 1::2] = mask_token
                # masked_target[..., 0::2] = mask_token
                # # assert all tokens for msb are unchanged and < 256, and all tokens for lsb are mask token
                # assert torch.all(masked_input[..., ::2] == input_tokens[..., ::2])
                # assert torch.all(masked_input[..., 1::2] == mask_token)
                # assert torch.all(masked_target[..., ::2] == mask_token)
                # assert torch.all(masked_target[..., 1::2] == target_tokens[..., 1::2])
                # assert torch.all(masked_input[..., ::2] < 256), (input_tokens[..., ::2], masked_input[..., ::2])
                # assert torch.all(masked_target[..., 1::2] < 256), (target_tokens[..., 1::2], masked_target[..., 1::2])

                masked_outputs = self(masked_input, labels=masked_input)
                # masked_loss = masked_outputs.loss
                # masked_bpb = masked_loss / np.log(2)
                # extract only the loss for the most significant bits
                shift_masked_logits = masked_outputs.logits[..., :-1, :].contiguous()
                shift_masked_labels = masked_input[..., 1:].contiguous()
                assert torch.all(shift_masked_labels[..., 1::2] < 256)
                assert torch.all(shift_masked_labels[..., ::2] == mask_token)
                masked_token_losses = loss_fct(shift_masked_logits.view(-1, shift_masked_logits.size(-1)), shift_masked_labels.view(-1))
                masked_token_losses = masked_token_losses.view(shift_masked_labels.shape)
                msb_token_losses = masked_token_losses[:, 1::2]  # take only the losses for the most significant bits
                masked_avg_loss = msb_token_losses.mean()
                masked_avg_bpb = masked_avg_loss / np.log(2)


                out_d['msb_loss'] = masked_avg_loss.detach().cpu()
                out_d['msb_bpb'] = masked_avg_bpb.detach().cpu()

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

        if self.hparams.get("lb_dropout", 0.0) > 0.0 and self.hparams.split_bit:
            # also log the msb-only loss/bpb
            msb_losses = []
            msb_bpbs = []
            for o in self.validation_outputs:
                if 'msb_loss' in o and 'msb_bpb' in o:
                    msb_losses.append(o['msb_loss'].numpy())
                    msb_bpbs.append(o['msb_bpb'].numpy())
            if len(msb_losses) > 0:
                mean_msb_loss = np.stack(msb_losses, axis=0).mean()
                mean_msb_bpb = np.stack(msb_bpbs, axis=0).mean()
                self.log('val/msb_loss', float(mean_msb_loss), on_epoch=True, prog_bar=True)
                self.log('val/msb_bpb', float(mean_msb_bpb), on_epoch=True, prog_bar=True)
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
                if not self.hparams.split_bit:
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
                    # plot 4 subplots: loss high bits, loss low bits, bpb high bits, bpb low bits
                    if self.hparams.split_bit == 2 or self.hparams.split_bit == True:
                        fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(2, 2, 1)
                        ax.plot(per_index_loss_arr[1::2], label='Per-index Loss High Bits', color='C0')
                        ax.set_title(f'{stage} Per-index Loss High Bits')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Loss (nats)')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 2)
                        ax.plot(per_index_loss_arr[::2], label='Per-index Loss Low Bits', color='C1')
                        ax.set_title(f'{stage} Per-index Loss Low Bits')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Loss (nats)')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 3)
                        ax.plot(per_index_bpb_arr[1::2], label='Per-index Bits-per-byte High Bits', color='C2')
                        ax.set_title(f'{stage} Per-index Bits-per-byte High Bits')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 4)
                        ax.plot(per_index_bpb_arr[::2], label='Per-index Bits-per-byte Low Bits', color='C3')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Low Bits')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        plt.tight_layout()
                        canvas.draw()
                        rgba = np.asarray(canvas.buffer_rgba())
                        im = Image.fromarray(rgba)
                        exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    elif self.hparams.split_bit == 4:
                        # do 2x2 grid of only bpb plots, for each of the 4 bit positions
                        fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(2, 2, 1)
                        ax.plot(per_index_bpb_arr[::4], label='Per-index Bits-per-byte Bit 3', color='C0')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Bit 3')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 2)
                        ax.plot(per_index_bpb_arr[1::4], label='Per-index Bits-per-byte Bit 2', color='C1')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Bit 2')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 3)
                        ax.plot(per_index_bpb_arr[2::4], label='Per-index Bits-per-byte Bit 1', color='C2')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Bit 1')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 4)
                        ax.plot(per_index_bpb_arr[3::4], label='Per-index Bits-per-byte Bit 0', color='C3')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Bit 0')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        plt.tight_layout()
                        canvas.draw()
                        rgba = np.asarray(canvas.buffer_rgba())
                        im = Image.fromarray(rgba)
                        exp.log({f'{stage}/per_index_plot': wandb.Image(im)}, step=step)
                    elif self.hparams.split_bit == 3:
                        # do first plot for 8-bit part, second plot for 4-bit part, third plot for last 4-bit part, and then 4th plot for all combined
                        fig = Figure(figsize=(8.29, 8.29), dpi=100, tight_layout=True)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(2, 2, 1)
                        ax.plot(per_index_bpb_arr[::3], label='Per-index Bits-per-byte Big 8-bit Part', color='C0')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Big 8-bit Part')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 2)
                        ax.plot(per_index_bpb_arr[1::3], label='Per-index Bits-per-byte Middle 4-bit Part', color='C1')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Middle 4-bit Part')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 3)
                        ax.plot(per_index_bpb_arr[2::3], label='Per-index Bits-per-byte Smallest 4-bit Part', color='C2')
                        ax.set_title(f'{stage} Per-index Bits-per-byte Smallest 4-bit Part')
                        ax.set_xlabel('Index (sample)')
                        ax.set_ylabel('Bits-per-byte')
                        ax.grid(True)
                        ax = fig.add_subplot(2, 2, 4)
                        ax.plot(per_index_bpb_arr, label='Per-index Bits-per-byte All Combined', color='C3')
                        ax.set_title(f'{stage} Per-index Bits-per-byte All Combined')
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
    def __init__(self, data_dir, batch_size=4, num_workers=4, chunk_size=1024, split_bit=False, only_lower_bits=False, train_p=1.0, stereo_interleave=False, lb_dropout=0.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.split_bit = split_bit
        self.only_lower_bits = only_lower_bits
        self.train_p = train_p
        self.stereo_interleave = stereo_interleave
        self.lb_dropout = lb_dropout

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_dir, 'train')
        valid_dir = os.path.join(self.data_dir, 'valid')

        # If explicit train/valid subdirectories exist, use them
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            train_full = MonoWavChunkDataset(train_dir, chunk_size=self.chunk_size, bit_split=self.split_bit, only_lower_bits=self.only_lower_bits, stereo_interleave=self.stereo_interleave, lb_dropout=self.lb_dropout)
            val_ds = MonoWavChunkDataset(valid_dir, chunk_size=self.chunk_size, bit_split=self.split_bit, only_lower_bits=self.only_lower_bits, stereo_interleave=self.stereo_interleave, lb_dropout=0.0)

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
            # Fallback: use single directory and split internally (original behavior)
            dataset = MonoWavChunkDataset(self.data_dir, chunk_size=self.chunk_size, bit_split=self.split_bit, only_lower_bits=self.only_lower_bits, stereo_interleave=self.stereo_interleave)
            n = len(dataset)
            n_train = int(0.9 * n)
            self.train_ds, self.val_ds, _ = torch.utils.data.random_split(dataset, [int(n_train * self.train_p), n - n_train, n_train - int(n_train * self.train_p)])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# =====================
# Training Entry Point
# =====================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=-1)
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
    args = parser.parse_args()

    # set seeds
    pl.seed_everything(args.seed)

    wandb_logger = WandbLogger(project=args.project)

    model = GPTAudioLightningModule(**vars(args))
    dm = MonoDataModule(args.data_dir, batch_size=args.batch_size, chunk_size=args.chunk_size, split_bit=args.split_bit, only_lower_bits=args.only_lower_bits, train_p=args.train_p, stereo_interleave=args.stereo_interleave, lb_dropout=args.lb_dropout)


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

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
