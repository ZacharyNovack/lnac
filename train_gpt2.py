import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import soundfile as sf
import torchaudio
import numpy as np
import wandb
from transformers import GPT2LMHeadModel, GPT2Config, get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import random
import json
from typing import Optional

# =====================
# Constants
# =====================
BYTES_PER_SAMPLE = 3
VOCAB_PER_BYTE = 256
MASK_TOKEN = BYTES_PER_SAMPLE * VOCAB_PER_BYTE  # 768
VOCAB_SIZE = MASK_TOKEN + 1  # 769


# =====================
# Mu-law encoding (for datasets that need it)
# =====================
def mu_law_encode(audio: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Mu-law companding: float [-1,1] -> unsigned int [0, 2^bits - 1]."""
    mu = (1 << bits) - 1
    audio = audio.clamp(-1.0, 1.0)
    numerator = torch.log1p(mu * torch.abs(audio))
    denominator = torch.log1p(torch.tensor(float(mu)))
    encoded = torch.sign(audio) * (numerator / denominator)
    encoded = ((encoded + 1) / 2 * mu).long().clamp(0, mu)
    return encoded


# =====================
# Audio loading
# =====================
def load_audio_raw(path: str, offset: int, num_frames: int, bits_per_sample: int) -> tuple[torch.Tensor, int]:
    """Load raw integer PCM via soundfile. Returns unsigned PCM as int64 tensor.

    soundfile scales sub-32-bit formats to fill int32 range.
    We right-shift to native bit depth, then add 2^(N-1) for signed->unsigned.
    """
    data, sr = sf.read(path, start=offset, stop=offset + num_frames, dtype='int32', always_2d=True)
    assert data.dtype == np.int32
    shift = 32 - bits_per_sample
    samples = torch.from_numpy(data.copy()).to(torch.int64)
    if shift > 0:
        samples = samples >> shift
    samples = samples + (1 << (bits_per_sample - 1))
    return samples.T, sr


def load_audio_mulaw(path: str, offset: int, num_frames: int, bits: int = 8) -> tuple[torch.Tensor, int]:
    """Load audio via torchaudio (float), apply mu-law encoding."""
    # wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames, backend="soundfile")
    wav, sr = sf.read(path, start=offset, stop=offset + num_frames, dtype='float32', always_2d=True)
    wav = torch.from_numpy(wav.copy()).to(torch.float32)
    return mu_law_encode(wav.clamp(-1.0, 1.0), bits=bits), sr


def split_to_bytes(samples: torch.Tensor, bits_per_sample: int,
                   bit_depth_cap: Optional[int] = None,
                   pad_to_max_bytes: bool = True) -> torch.Tensor:
    """Split unsigned PCM into interleaved byte tokens with sub-vocabulary offsets.

    With pad_to_max_bytes=True (default), always pads to 3 bytes/sample:
        24-bit: [MSB+0, MID+256, LSB+512]
        16-bit: [MSB+0, LSB+256, MASK]
        8-bit:  [MSB+0, MASK, MASK]
    With pad_to_max_bytes=False, uses native byte count (no MASK padding).
    """
    effective_bits = bits_per_sample
    if bit_depth_cap is not None:
        effective_bits = min(effective_bits, bit_depth_cap)
        if effective_bits < bits_per_sample:
            shift = bits_per_sample - effective_bits
            samples = (samples >> shift) << shift

    n_bytes = (effective_bits + 7) // 8

    byte_list = []
    for i in range(n_bytes):
        shift = (bits_per_sample - 8) - (i * 8)
        if shift >= 0:
            byte_val = (samples >> shift) & 0xFF
        else:
            byte_val = (samples << (-shift)) & 0xFF
        byte_list.append(byte_val + i * VOCAB_PER_BYTE)

    if pad_to_max_bytes:
        while len(byte_list) < BYTES_PER_SAMPLE:
            byte_list.append(torch.full_like(samples, MASK_TOKEN))

    return torch.stack(byte_list, dim=-1).reshape(-1)


# =====================
# Dataset
# =====================
class AudioByteDataset(Dataset):
    """Loads raw audio as byte-level tokens for autoregressive LM training.

    Each sample -> 3 interleaved byte tokens with per-position sub-vocabularies.
    Lower bit-depth audio gets MASK tokens for missing bytes.
    """

    def __init__(self, data_dir: str, metadata_path: str, chunk_size: int = 1024,
                 encoding: str = 'linear', bit_depth_cap: Optional[int] = None,
                 stereo_interleave: bool = False, epoch_expansion_factor: int = 1,
                 pad_to_max_bytes: bool = True):
        metadata = json.load(open(metadata_path, 'r'))
        all_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.endswith(('.flac', '.wav'))
        )
        self.files = [(f, metadata[os.path.basename(f)]) for f in all_files if os.path.basename(f) in metadata]
        self.chunk_size = chunk_size
        self.encoding = encoding
        self.bit_depth_cap = bit_depth_cap
        self.stereo_interleave = stereo_interleave
        self.pad_to_max_bytes = pad_to_max_bytes

        self.files = [(f, m) for f, m in self.files
                      if m['length'] >= self._audio_chunk_for(m['bits_per_sample']) + 1]
        if not self.files:
            raise ValueError(f"No valid files in {data_dir} with metadata {metadata_path}")

        if epoch_expansion_factor > 1:
            self.files = self.files * epoch_expansion_factor
        random.shuffle(self.files)
        print(f"AudioByteDataset: {len(self.files)} entries, chunk={chunk_size}, enc={encoding}")

    def _audio_chunk_for(self, bits_per_sample: int) -> int:
        """How many audio samples to read for a given bit depth.

        When pad_to_max_bytes=True, always read chunk_size samples (pad short bytes with MASK).
        When False, read more samples for lower bit depths so total tokens = chunk_size * 3.
        """
        if self.pad_to_max_bytes:
            return self.chunk_size
        effective = min(bits_per_sample, self.bit_depth_cap) if self.bit_depth_cap else bits_per_sample
        n_bytes = (effective + 7) // 8
        return self.chunk_size * BYTES_PER_SAMPLE // n_bytes

    @property
    def effective_n_bytes(self) -> int:
        """Bytes per sample for this dataset."""
        if self.pad_to_max_bytes:
            return BYTES_PER_SAMPLE
        _, meta = self.files[0]
        bits = meta['bits_per_sample']
        effective = min(bits, self.bit_depth_cap) if self.bit_depth_cap else bits
        return (effective + 7) // 8

    @property
    def seq_len(self) -> int:
        """Sequence length (always chunk_size * 3 * stereo_mult + 1)."""
        stereo_mult = 2 if self.stereo_interleave else 1
        return self.chunk_size * BYTES_PER_SAMPLE * stereo_mult + 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path, meta = self.files[idx]
            bits = meta['bits_per_sample']
            file_chunk = self._audio_chunk_for(bits)
            offset = random.randint(0, max(1, meta['length'] - file_chunk - 1))
            num_frames = file_chunk + 1

            if self.encoding == 'mu-law':
                effective = self.bit_depth_cap or bits
                wav, _ = load_audio_mulaw(path, offset, num_frames, bits=effective)
                bits = effective
            else:
                wav, _ = load_audio_raw(path, offset, num_frames, bits)

            # Channel selection
            if self.stereo_interleave:
                if wav.shape[0] < 2:
                    wav = torch.cat([wav, wav], dim=0)
                order = [0, 1] if random.random() < 0.5 else [1, 0]
                wav = torch.cat([wav[order[0]], wav[order[1]]], dim=0)
            else:
                wav = wav[random.randint(0, 1)] if wav.shape[0] == 2 else wav[0]

            tokens = split_to_bytes(wav, bits, self.bit_depth_cap, self.pad_to_max_bytes)

            if len(tokens) < self.seq_len:
                tokens = torch.nn.functional.pad(tokens, (0, self.seq_len - len(tokens)), value=MASK_TOKEN)

            return tokens[:self.seq_len].clone()

        except Exception as e:
            print(f"Error loading {self.files[idx][0]}: {e}")
            return self[(idx + random.randint(1, len(self.files) - 1)) % len(self.files)]


# =====================
# Lightning Module
# =====================
class GPTAudioLightningModule(pl.LightningModule):
    def __init__(self, model_name='gpt2', lr=3e-4, weight_decay=0.1,
                 warmup_steps=1000, max_steps=-1, chunk_size=1024,
                 stereo_interleave=False, pad_to_max_bytes=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = GPT2Config.from_pretrained(model_name)
        config.vocab_size = VOCAB_SIZE  # always 769 (superset works for all byte counts)
        stereo_mult = 2 if stereo_interleave else 1
        config.max_position_embeddings = chunk_size * BYTES_PER_SAMPLE * stereo_mult + 1
        self.model = GPT2LMHeadModel(config)
        if kwargs.get('gradient_checkpointing'):
            self.model.gradient_checkpointing_enable()
        self.n_bytes = BYTES_PER_SAMPLE  # for per-byte logging

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)

    def _per_byte_bpb(self, batch, logits, n_bytes=None):
        """Compute mean bpb for each byte position."""
        if n_bytes is None:
            n_bytes = self.n_bytes
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        token_bpb = nn.CrossEntropyLoss(reduction='none')(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape).mean(0) / np.log(2)
        labels = ['MSB', 'MID', 'LSB'][:n_bytes]
        return {label: float(token_bpb[i::n_bytes].mean())
                for i, label in enumerate(labels)}

    def training_step(self, batch, batch_idx):
        outputs = self(batch, labels=batch)
        bpb = outputs.loss / np.log(2)
        self.log('train/loss', outputs.loss, on_step=True, on_epoch=True)
        self.log('train/bpb', bpb, on_step=True, on_epoch=True, prog_bar=True)

        if self.global_step % 500 == 0:
            with torch.no_grad():
                for label, val in self._per_byte_bpb(batch, outputs.logits).items():
                    self.log(f'train/bpb_{label}', val, on_step=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch, labels=batch)
        bpb = outputs.loss / np.log(2)
        dm = self.trainer.datamodule
        n_bytes = dm.val_n_bytes[dataloader_idx]
        per_byte = self._per_byte_bpb(batch, outputs.logits, n_bytes)
        self.validation_outputs[dataloader_idx].append({'bpb': float(bpb), **per_byte})

    def on_validation_epoch_start(self):
        dm = self.trainer.datamodule
        self.validation_outputs = {i: [] for i in range(len(dm.val_dataset_names))}

    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        all_bpb = []
        for i, name in enumerate(dm.val_dataset_names):
            outputs = self.validation_outputs.get(i, [])
            if not outputs:
                continue
            mean_bpb = np.mean([o['bpb'] for o in outputs])
            self.log(f'val/bpb_{name}', mean_bpb, on_epoch=True)
            all_bpb.extend([o['bpb'] for o in outputs])
            for label in ['MSB', 'MID', 'LSB'][:dm.val_n_bytes[i]]:
                self.log(f'val/bpb_{name}_{label}',
                         np.mean([o[label] for o in outputs]), on_epoch=True)
        if all_bpb:
            self.log('val/bpb', np.mean(all_bpb), on_epoch=True, prog_bar=True)
        self.validation_outputs = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=self.hparams.weight_decay,
        )
        total_steps = (self.hparams.max_steps if self.hparams.max_steps > 0
                       else self.trainer.estimated_stepping_batches)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=total_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


# =====================
# Data Module
# =====================
class AudioDataModule(pl.LightningDataModule):
    """Reads a JSON config with pre-split train/val datasets.

    Config: {"datasets": {"name": {"train_data_dir", "val_data_dir",
             "train_metadata_path", "val_metadata_path", "stereo_interleave",
             "encoding", "bit_depth_cap", "epoch_expansion_factor"}, ...}}
    """

    def __init__(self, dataset_config_path: str, batch_size: int = 8,
                 num_workers: int = 4, chunk_size: int = 1024,
                 epoch_expansion_factor: int = 1, pad_to_max_bytes: bool = True):
        super().__init__()
        self.dataset_config_path = dataset_config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.epoch_expansion_factor = epoch_expansion_factor
        self.pad_to_max_bytes = pad_to_max_bytes

    def _build_dataset(self, data_dir: str, metadata_path: str, ds_cfg: dict) -> AudioByteDataset:
        stereo = ds_cfg.get('stereo_interleave', False)
        return AudioByteDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            chunk_size=self.chunk_size if stereo else self.chunk_size * 2,
            encoding=ds_cfg.get('encoding', 'linear'),
            bit_depth_cap=ds_cfg.get('bit_depth_cap', None),
            stereo_interleave=stereo,
            epoch_expansion_factor=ds_cfg.get('epoch_expansion_factor', self.epoch_expansion_factor),
            pad_to_max_bytes=self.pad_to_max_bytes,
        )

    def setup(self, stage=None):
        config = json.load(open(self.dataset_config_path))
        train_datasets = []
        self.val_datasets = []
        self.val_dataset_names = []
        self.val_n_bytes = []
        for name, ds_cfg in config['datasets'].items():
            print(f"[AudioDataModule] Setting up '{name}'")
            train_datasets.append(self._build_dataset(
                ds_cfg['train_data_dir'], ds_cfg['train_metadata_path'], ds_cfg))
            val_ds = self._build_dataset(
                ds_cfg['val_data_dir'], ds_cfg['val_metadata_path'], ds_cfg)
            self.val_datasets.append(val_ds)
            self.val_dataset_names.append(name)
            self.val_n_bytes.append(val_ds.effective_n_bytes)
        self.train_ds = torch.utils.data.ConcatDataset(train_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        val_workers = max(1, self.num_workers // max(1, len(self.val_datasets)))
        return [DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                           num_workers=val_workers, pin_memory=True,
                           persistent_workers=val_workers > 0)
                for ds in self.val_datasets]


# =====================
# Entry Point
# =====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GPT-2 on byte-level audio')
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=500_000)
    parser.add_argument('--project', type=str, default='lnac_redux')
    parser.add_argument('--no_pad_to_max_bytes', dest='pad_to_max_bytes',
                        action='store_false', default=True,
                        help='Use native byte count per sample instead of padding to 3 with MASK')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--epoch_expansion_factor', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Validate every N steps (int) or fraction of epoch (float)')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to reduce memory')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Accumulate gradients over N batches')
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project=args.project)

    dataset_config = json.load(open(args.dataset_config))
    has_stereo = any(ds.get('stereo_interleave', False)
                     for ds in dataset_config['datasets'].values())

    model = GPTAudioLightningModule(
        model_name=args.model_name, lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, max_steps=args.max_steps,
        chunk_size=args.chunk_size, stereo_interleave=has_stereo,
        pad_to_max_bytes=args.pad_to_max_bytes,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.compile:
        model.model = torch.compile(model.model)

    dm = AudioDataModule(
        dataset_config_path=args.dataset_config, batch_size=args.batch_size,
        num_workers=args.num_workers, chunk_size=args.chunk_size,
        epoch_expansion_factor=args.epoch_expansion_factor,
        pad_to_max_bytes=args.pad_to_max_bytes,
    )

    val_check = int(args.val_check_interval) if args.val_check_interval >= 1 else args.val_check_interval

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        max_epochs=args.max_epochs, max_steps=args.max_steps,
        precision='bf16-mixed', logger=wandb_logger,
        callbacks=[ModelCheckpoint(monitor="val/bpb", mode="min", save_top_k=1,
                                   save_last=True, every_n_epochs=10,
                                   filename="gpt2audio-{epoch:02d}")],
        gradient_clip_val=1.0, log_every_n_steps=50,
        val_check_interval=val_check,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
