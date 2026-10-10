from __future__ import annotations

import os
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eeg.models.atcnet import Model as _ATCNetImpl
from eeg.models.bendr import Model as _BENDRImpl
from eeg.models.biot import Model as _BIOTImpl
from eeg.models.cbramod import Model as _CBraModImpl
from eeg.models.deepconvnet import Model as _DeepConvNetImpl
from eeg.models.dbconformer import Model as _DBConformerImpl
from eeg.models.eeg_deformer import Model as _EEGDeformerImpl
from eeg.models.eegconformer import Model as _EEGConformerImpl
from eeg.models.eegnet import Model as _EEGNetImpl
from eeg.models.eegpt import Model as _EEGPTImpl
from eeg.models.fbcnet import Model as _FBCNetImpl
from eeg.models.labram import Model as _LaBraMImpl
from eeg.models.lmda_net import Model as _LMDANetImpl
from eeg.models.shallowconvnet import Model as _ShallowConvNetImpl
from eeg.models.tsception import Model as _TSceptionImpl


AVAILABLE_MODELS = [
    "DeepConvNet",
    "ShallowConvNet",
    "EEGNet",
    "FBCNet",
    "TSception",
    "ATCNet",
    "EEG-Conformer",
    "LMDA-Net",
    "EEG-Deformer",
    "DBConformer",
    "BENDR",
    "BIOT",
    "LaBraM",
    "EEGPT",
    "CBraMod",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_heads(embed_dim: int) -> int:
    for h in (8, 4, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pick_array(container: np.lib.npyio.NpzFile, keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in container:
            return container[key]
    raise KeyError(f"None of these keys were found: {keys}")


def load_npz_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    x = _pick_array(data, ("X", "x", "data", "signals"))
    y = _pick_array(data, ("y", "Y", "labels", "label"))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 3:
        raise ValueError(f"{path} must contain X with shape [N, C, T], got {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"{path} X and y lengths do not match: {len(x)} vs {len(y)}")
    return x, y


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - val_ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def normalize_train_stats(
    train_x: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, ...]:
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True) + eps
    out = [(train_x - mean) / std]
    for item in others:
        out.append((item - mean) / std)
    return tuple(out)


class EEGAugmentor:
    def __init__(
        self,
        max_shift_ratio: float = 0.08,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        channel_drop_prob: float = 0.05,
        time_mask_ratio: float = 0.06,
    ):
        self.max_shift_ratio = max_shift_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.channel_drop_prob = channel_drop_prob
        self.time_mask_ratio = time_mask_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, copy=True)
        c, t = out.shape
        shift = int(round(t * self.max_shift_ratio))
        if shift > 0:
            offset = np.random.randint(-shift, shift + 1)
            out = np.roll(out, offset, axis=1)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        out *= scale
        if self.noise_std > 0:
            out += np.random.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
        if self.channel_drop_prob > 0:
            mask = np.random.rand(c) < self.channel_drop_prob
            out[mask] = 0.0
        mask_len = int(round(t * self.time_mask_ratio))
        if mask_len > 0 and t > mask_len:
            start = np.random.randint(0, t - mask_len + 1)
            out[:, start : start + mask_len] = 0.0
        return out.astype(np.float32, copy=False)


def _read_numeric_vector(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _read_signal_matrix(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.asarray(arr, dtype=np.float32)


def _trial_window_seconds(signal_path: Path, num_channels: int) -> Tuple[float, float, int]:
    name = signal_path.stem.upper()
    if num_channels >= 60:
        return 3.0, 7.0, 250
    if num_channels == 2:
        if "O3VR" in name:
            return 4.0, 8.0, 125
        return 4.0, 7.0, 125
    raise ValueError(f"Unsupported channel count: {num_channels}")


def _collect_competition_trials(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_files = sorted(data_dir.glob("*_s.txt"))
    if not signal_files:
        raise FileNotFoundError(f"No *_s.txt files found in {data_dir}")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    arts: List[np.ndarray] = []
    for signal_path in signal_files:
        prefix = signal_path.stem[:-2]
        trig_path = data_dir / f"{prefix}_HDR_TRIG.txt"
        label_path = data_dir / f"{prefix}_HDR_Classlabel.txt"
        art_path = data_dir / f"{prefix}_HDR_ArtifactSelection.txt"
        if not trig_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing header files for {signal_path.name}")

        signals = _read_signal_matrix(signal_path)
        trig = _read_numeric_vector(trig_path).astype(np.int64)
        labels = _read_numeric_vector(label_path)
        artifacts = _read_numeric_vector(art_path).astype(np.int64) if art_path.exists() else np.zeros_like(labels, dtype=np.int64)

        start_s, end_s, fs = _trial_window_seconds(signal_path, signals.shape[1])
        start_offset = int(round(start_s * fs))
        window_len = int(round((end_s - start_s) * fs))

        trial_x: List[np.ndarray] = []
        trial_y: List[int] = []
        trial_art: List[int] = []
        for start, label, art in zip(trig, labels, artifacts):
            if math.isnan(float(label)):
                continue
            begin = int(start) + start_offset
            end = begin + window_len
            if end > len(signals):
                continue
            seg = signals[begin:end]
            if np.isnan(seg).any():
                seg = np.nan_to_num(seg, nan=0.0)
            trial_x.append(seg.T)
            trial_y.append(int(label) - 1)
            trial_art.append(int(art))

        if trial_x:
            xs.append(np.stack(trial_x, axis=0).astype(np.float32))
            ys.append(np.asarray(trial_y, dtype=np.int64))
            arts.append(np.asarray(trial_art, dtype=np.int64))

    if not xs:
        raise RuntimeError(f"No labeled trials could be parsed from {data_dir}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(arts)


def _stratified_split_indices(y: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val :].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def prepare_competition_iii_npz(data_dir: Path, seed: int = 42, drop_artifacts: bool = True) -> None:
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    test_path = data_dir / "test.npz"
    if train_path.exists() and val_path.exists() and test_path.exists():
        return
    if not list(data_dir.glob("*_s.txt")):
        return

    x, y, artifacts = _collect_competition_trials(data_dir)
    raw_trials = len(x)
    if drop_artifacts:
        keep = artifacts == 0
        if keep.any():
            x = x[keep]
            y = y[keep]

    train_idx, val_idx, test_idx = _stratified_split_indices(y, train_ratio=0.7, val_ratio=0.15, seed=seed)
    np.savez_compressed(train_path, X=x[train_idx], y=y[train_idx])
    np.savez_compressed(val_path, X=x[val_idx], y=y[val_idx])
    np.savez_compressed(test_path, X=x[test_idx], y=y[test_idx])

    report = {
        "raw_labeled_trials": int(raw_trials),
        "used_trials": int(len(x)),
        "train_trials": int(len(train_idx)),
        "val_trials": int(len(val_idx)),
        "test_trials": int(len(test_idx)),
        "shape": list(x.shape),
        "classes_zero_based": sorted(np.unique(y).astype(int).tolist()),
        "drop_artifacts": bool(drop_artifacts),
    }
    with (data_dir / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


class EEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: Optional[EEGAugmentor] = None):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[index].numpy()
        if self.augment is not None:
            sample = self.augment(sample)
        return torch.from_numpy(sample), self.y[index]


def load_dataset_bundle(data_dir: Path, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prepare_competition_iii_npz(data_dir, seed=seed)
    train_path = data_dir / "train.npz"
    test_path = data_dir / "test.npz"
    val_path = data_dir / "val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    train_x, train_y = load_npz_file(train_path)
    test_x, test_y = load_npz_file(test_path)

    if val_path.exists():
        val_x, val_y = load_npz_file(val_path)
    else:
        train_x, train_y, val_x, val_y = train_val_split(train_x, train_y, val_ratio=0.2, seed=seed)

    train_x, val_x, test_x = normalize_train_stats(train_x, val_x, test_x)
    return train_x, train_y, val_x, val_y, test_x, test_y


def build_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_augment = EEGAugmentor() if augment else None
    train_loader = DataLoader(
        EEGDataset(train_x, train_y, augment=train_augment),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EEGDataset(val_x, val_y, augment=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        EEGDataset(test_x, test_y, augment=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def _is_iii_a(in_chans: int) -> bool:
    return int(in_chans) >= 60


def _is_long_trial(time_steps: int) -> bool:
    return int(time_steps) >= 500


def DeepConvNet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    dropout = 0.45 if _is_iii_a(in_chans) else 0.35
    return _DeepConvNetImpl(channels=in_chans, samples=time_steps, num_classes=num_classes, dropout=dropout)


def ShallowConvNet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    dropout = 0.5 if _is_iii_a(in_chans) else 0.45
    return _ShallowConvNetImpl(channels=in_chans, samples=time_steps, num_classes=num_classes, dropout=dropout)


def EEGNet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    dropout = 0.5 if _is_iii_a(in_chans) else 0.4
    return _EEGNetImpl(num_classes=num_classes, channels=in_chans, samples=time_steps, dropout=dropout)


def FBCNet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _FBCNetImpl(
        channels=in_chans,
        samples=time_steps,
        num_classes=num_classes,
        n_bands=9 if _is_iii_a(in_chans) else 5,
        n_filters_spat=32 if _is_iii_a(in_chans) else 24,
        dropout=0.45 if _is_iii_a(in_chans) else 0.4,
    )


def TSception(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    sampling_rate = 250 if _is_long_trial(time_steps) else 125
    return _TSceptionImpl(
        channels=in_chans,
        samples=time_steps,
        num_classes=num_classes,
        num_T=18 if _is_iii_a(in_chans) else 12,
        num_S=18 if _is_iii_a(in_chans) else 12,
        hid_channels=48 if _is_iii_a(in_chans) else 32,
        dropout=0.5 if _is_iii_a(in_chans) else 0.45,
        sampling_rate=sampling_rate,
    )


def ATCNet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _ATCNetImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        n_filters_time=8 if _is_iii_a(in_chans) else 6,
        n_filters_spat=16 if _is_iii_a(in_chans) else 12,
        filter_time_length=64 if _is_long_trial(time_steps) else 32,
        pool_time_length=8,
        pool_time_stride=4,
        n_heads=4,
        n_blocks=3 if _is_iii_a(in_chans) else 2,
        dropout=0.5,
    )


def EEGConformer(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _EEGConformerImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        n_filters_time=40 if _is_iii_a(in_chans) else 24,
        filter_time_length=25 if _is_long_trial(time_steps) else 15,
        pool_time_length=75 if _is_long_trial(time_steps) else 37,
        pool_time_stride=15 if _is_long_trial(time_steps) else 7,
        n_filters_spat=40 if _is_iii_a(in_chans) else 24,
        att_depth=6 if _is_iii_a(in_chans) else 4,
        att_heads=10 if _is_iii_a(in_chans) else 8,
        drop_prob=0.5 if _is_iii_a(in_chans) else 0.35,
    )


def LMDANet(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _LMDANetImpl(
        chans=in_chans,
        samples=time_steps,
        num_classes=num_classes,
        depth=9 if _is_iii_a(in_chans) else 6,
        kernel=75 if _is_long_trial(time_steps) else 31,
        channel_depth1=24 if _is_iii_a(in_chans) else 16,
        channel_depth2=9 if _is_iii_a(in_chans) else 6,
        ave_depth=1,
        avepool=5 if _is_iii_a(in_chans) else 3,
    )


def EEGDeformer(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _EEGDeformerImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        emb_size=40 if _is_iii_a(in_chans) else 32,
        n_heads=4,
        n_hct_layers=2 if _is_iii_a(in_chans) else 3,
        fft_size=128 if _is_iii_a(in_chans) else 96,
        dropout=0.12 if _is_iii_a(in_chans) else 0.15,
    )


def DBConformer(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _DBConformerImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        emb_size=48 if _is_iii_a(in_chans) else 32,
        n_heads=4,
        n_layers=3 if _is_iii_a(in_chans) else 2,
        ffn_size=192 if _is_iii_a(in_chans) else 128,
        dropout=0.3,
    )


def BENDR(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _BENDRImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        encoder_h=240 if _is_iii_a(in_chans) else 256,
        contextualizer_hidden=960 if _is_iii_a(in_chans) else 1024,
        transformer_layers=6 if _is_iii_a(in_chans) else 4,
        transformer_heads=8,
        drop_prob=0.1 if _is_iii_a(in_chans) else 0.15,
        position_encoder_length=25,
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(3, 2, 2, 2, 2, 2),
        start_token=-5.0,
        final_layer=True,
    )


def BIOT(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _BIOTImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        patch_len=250 if _is_iii_a(in_chans) else 125,
        stride=50 if _is_iii_a(in_chans) else 25,
        embed_dim=128 if _is_iii_a(in_chans) else 96,
        num_heads=8 if _is_iii_a(in_chans) else 4,
        num_layers=4 if _is_iii_a(in_chans) else 3,
        mlp_ratio=4.0,
        drop_prob=0.2,
    )


def LaBraM(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _LaBraMImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        patch_size=100 if _is_iii_a(in_chans) else 75,
        embed_dim=128 if _is_iii_a(in_chans) else 96,
        depth=6 if _is_iii_a(in_chans) else 4,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.05,
        drop_path_rate=0.05,
        use_channel_embed=True,
        use_pos_embed=True,
        init_values=0.1,
    )


def EEGPT(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _EEGPTImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        patch_size=100 if _is_iii_a(in_chans) else 75,
        embed_dim=192 if _is_iii_a(in_chans) else 128,
        embed_num=4,
        depth=6 if _is_iii_a(in_chans) else 4,
        num_heads=8 if _is_iii_a(in_chans) else 4,
        mlp_ratio=4.0,
        drop_prob=0.15,
    )


def CBraMod(in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    return _CBraModImpl(
        n_chans=in_chans,
        n_samples=time_steps,
        n_outputs=num_classes,
        patch_size=100 if _is_iii_a(in_chans) else 75,
        embed_dim=128 if _is_iii_a(in_chans) else 96,
        num_heads=8 if _is_iii_a(in_chans) else 6,
        num_layers=8 if _is_iii_a(in_chans) else 6,
        mlp_ratio=4.0,
        dropout=0.1,
        return_encoder_output=False,
    )


MODEL_BUILDERS = {
    "DeepConvNet": DeepConvNet,
    "ShallowConvNet": ShallowConvNet,
    "EEGNet": EEGNet,
    "FBCNet": FBCNet,
    "TSception": TSception,
    "ATCNet": ATCNet,
    "EEG-Conformer": EEGConformer,
    "EEGConformer": EEGConformer,
    "LMDA-Net": LMDANet,
    "LMDANet": LMDANet,
    "EEG-Deformer": EEGDeformer,
    "EEGDeformer": EEGDeformer,
    "DBConformer": DBConformer,
    "BENDR": BENDR,
    "BIOT": BIOT,
    "LaBraM": LaBraM,
    "EEGPT": EEGPT,
    "CBraMod": CBraMod,
}


def build_model(model_name: str, in_chans: int, num_classes: int, time_steps: int) -> nn.Module:
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {model_name}. Available: {', '.join(AVAILABLE_MODELS)}")
    return MODEL_BUILDERS[model_name](in_chans, num_classes, time_steps)


@dataclass
class EpochResult:
    loss: float
    acc: float


@dataclass
class EvalResult:
    loss: float
    acc: float
    kappa: float
    recall: float
    f1: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
    return EpochResult(loss=total_loss / max(1, total_samples), acc=total_correct / max(1, total_samples))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float, float, float]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = float(correct / total) if total else 0.0

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = float(cm[cls, cls])
        fn = float(row_sum[cls] - tp)
        fp = float(col_sum[cls] - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1s.append(f1)

    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    pe = float(np.dot(row_sum, col_sum)) / float(total * total) if total else 0.0
    kappa = float((acc - pe) / (1.0 - pe)) if (1.0 - pe) else 0.0
    return acc, kappa, macro_recall, macro_f1


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    acc, kappa, recall, f1 = classification_metrics(y_true_arr, y_pred_arr, num_classes)
    return EvalResult(
        loss=total_loss / max(1, total_samples),
        acc=acc,
        kappa=kappa,
        recall=recall,
        f1=f1,
    )


def resolve_training_hparams(
    data_dir: Path,
    model_name: str,
    in_chans: int,
    time_steps: int,
) -> Dict[str, float]:
    is_iii_a = in_chans >= 60
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {
        "DeepConvNet": {
            "iii_a": {"epochs": 180, "batch_size": 12, "lr": 6e-4, "weight_decay": 5e-4, "label_smoothing": 0.04, "patience": 24, "augment": 1.0},
            "iii_b": {"epochs": 140, "batch_size": 16, "lr": 8e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 18, "augment": 1.0},
        },
        "ShallowConvNet": {
            "iii_a": {"epochs": 200, "batch_size": 12, "lr": 4e-4, "weight_decay": 5e-4, "label_smoothing": 0.05, "patience": 28, "augment": 1.0},
            "iii_b": {"epochs": 160, "batch_size": 16, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.04, "patience": 22, "augment": 1.0},
        },
        "EEGNet": {
            "iii_a": {"epochs": 180, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.03, "patience": 22, "augment": 1.0},
            "iii_b": {"epochs": 140, "batch_size": 32, "lr": 1e-3, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 18, "augment": 1.0},
        },
        "FBCNet": {
            "iii_a": {"epochs": 200, "batch_size": 12, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.04, "patience": 28, "augment": 1.0},
            "iii_b": {"epochs": 160, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.03, "patience": 20, "augment": 1.0},
        },
        "TSception": {
            "iii_a": {"epochs": 220, "batch_size": 12, "lr": 5e-4, "weight_decay": 5e-4, "label_smoothing": 0.05, "patience": 30, "augment": 1.0},
            "iii_b": {"epochs": 180, "batch_size": 16, "lr": 7e-4, "weight_decay": 4e-4, "label_smoothing": 0.04, "patience": 24, "augment": 1.0},
        },
        "ATCNet": {
            "iii_a": {"epochs": 220, "batch_size": 8, "lr": 8e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 26, "augment": 1.0},
            "iii_b": {"epochs": 180, "batch_size": 16, "lr": 1e-3, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 20, "augment": 1.0},
        },
        "EEG-Conformer": {
            "iii_a": {"epochs": 240, "batch_size": 8, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 30, "augment": 1.0},
            "iii_b": {"epochs": 200, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 24, "augment": 1.0},
        },
        "LMDA-Net": {
            "iii_a": {"epochs": 180, "batch_size": 12, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.04, "patience": 24, "augment": 1.0},
            "iii_b": {"epochs": 140, "batch_size": 16, "lr": 1e-3, "weight_decay": 3e-4, "label_smoothing": 0.03, "patience": 18, "augment": 1.0},
        },
        "EEG-Deformer": {
            "iii_a": {"epochs": 240, "batch_size": 8, "lr": 5e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 32, "augment": 1.0},
            "iii_b": {"epochs": 200, "batch_size": 16, "lr": 7e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 24, "augment": 1.0},
        },
        "DBConformer": {
            "iii_a": {"epochs": 220, "batch_size": 8, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 28, "augment": 1.0},
            "iii_b": {"epochs": 180, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 22, "augment": 1.0},
        },
        "BENDR": {
            "iii_a": {"epochs": 220, "batch_size": 8, "lr": 5e-4, "weight_decay": 5e-4, "label_smoothing": 0.02, "patience": 32, "augment": 1.0},
            "iii_b": {"epochs": 180, "batch_size": 16, "lr": 7e-4, "weight_decay": 4e-4, "label_smoothing": 0.02, "patience": 24, "augment": 1.0},
        },
        "BIOT": {
            "iii_a": {"epochs": 200, "batch_size": 8, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 28, "augment": 1.0},
            "iii_b": {"epochs": 160, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 20, "augment": 1.0},
        },
        "LaBraM": {
            "iii_a": {"epochs": 240, "batch_size": 8, "lr": 5e-4, "weight_decay": 4e-4, "label_smoothing": 0.02, "patience": 30, "augment": 1.0},
            "iii_b": {"epochs": 200, "batch_size": 16, "lr": 7e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 24, "augment": 1.0},
        },
        "EEGPT": {
            "iii_a": {"epochs": 240, "batch_size": 8, "lr": 4e-4, "weight_decay": 4e-4, "label_smoothing": 0.02, "patience": 32, "augment": 1.0},
            "iii_b": {"epochs": 200, "batch_size": 16, "lr": 6e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 24, "augment": 1.0},
        },
        "CBraMod": {
            "iii_a": {"epochs": 220, "batch_size": 8, "lr": 6e-4, "weight_decay": 4e-4, "label_smoothing": 0.03, "patience": 28, "augment": 1.0},
            "iii_b": {"epochs": 180, "batch_size": 16, "lr": 8e-4, "weight_decay": 3e-4, "label_smoothing": 0.02, "patience": 22, "augment": 1.0},
        },
    }
    group = profiles.get(model_name)
    if group is None:
        return {
            "epochs": 160 if is_iii_a else 140,
            "batch_size": 12 if is_iii_a else 16,
            "lr": 6e-4 if is_iii_a else 8e-4,
            "weight_decay": 4e-4,
            "label_smoothing": 0.03,
            "patience": 24 if is_iii_a else 20,
            "augment": 1.0,
        }
    return group["iii_a" if is_iii_a else "iii_b"]


def save_history(history: Dict[str, List[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        epochs = len(history["train_loss"])
        for idx in range(epochs):
            writer.writerow(
                [
                    idx + 1,
                    history["train_loss"][idx],
                    history["train_acc"][idx],
                    history["val_loss"][idx],
                    history["val_acc"][idx],
                ]
            )


def plot_history(history: Dict[str, List[float]], save_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def train_from_folder(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    patience: Optional[int] = None,
    seed: int = 42,
    device: Optional[str] = None,
) -> Path:
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset_bundle(data_dir, seed=seed)
    in_chans, time_steps = train_x.shape[1], train_x.shape[2]
    num_classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)

    hparams = resolve_training_hparams(data_dir, model_name, in_chans, time_steps)
    epochs = int(epochs or hparams["epochs"])
    batch_size = int(batch_size or hparams["batch_size"])
    lr = float(lr or hparams["lr"])
    weight_decay = float(weight_decay or hparams["weight_decay"])
    label_smoothing = float(label_smoothing if label_smoothing is not None else hparams["label_smoothing"])
    patience = int(patience or hparams["patience"])

    train_loader, val_loader, test_loader = build_loaders(
        train_x, train_y, val_x, val_y, test_x, test_y, batch_size=batch_size, augment=True
    )

    model = build_model(model_name, in_chans, num_classes, time_steps).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)

    run_dir = ensure_dir(output_dir / model_name)
    best_path = run_dir / "best_model.pt"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val = -1.0
    stale_epochs = 0

    for epoch in range(epochs):
        train_res = run_epoch(model, train_loader, criterion, optimizer, device)
        val_res = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_res.acc)
        history["train_loss"].append(train_res.loss)
        history["train_acc"].append(train_res.acc)
        history["val_loss"].append(val_res.loss)
        history["val_acc"].append(val_res.acc)
        if val_res.acc > best_val:
            best_val = val_res.acc
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
        print(
            f"[{model_name}] epoch {epoch + 1:03d}/{epochs} "
            f"train_loss={train_res.loss:.4f} train_acc={train_res.acc:.4f} "
            f"val_loss={val_res.loss:.4f} val_acc={val_res.acc:.4f}"
        )
        if stale_epochs >= patience:
            print(f"[{model_name}] early stopping at epoch {epoch + 1}")
            break

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint)
    test_res = evaluate_model(model, test_loader, criterion, device, num_classes)

    save_history(history, run_dir / "history.csv")
    plot_history(history, run_dir / "training_curve.png", f"{data_dir.name} - {model_name}")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "best_val_acc": best_val,
                "test_loss": test_res.loss,
                "test_acc": test_res.acc,
                "test_kappa": test_res.kappa,
                "test_recall": test_res.recall,
                "test_f1": test_res.f1,
                "in_chans": in_chans,
                "time_steps": time_steps,
                "num_classes": num_classes,
            },
            f,
            indent=2,
        )

    print(
        f"[{model_name}] "
        f"acc={test_res.acc:.4f} "
        f"kappa={test_res.kappa:.4f} "
        f"recall={test_res.recall:.4f} "
        f"f1={test_res.f1:.4f} "
        f"saved_to={run_dir}"
    )
    return run_dir
