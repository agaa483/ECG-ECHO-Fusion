"""
data_loader.py — Load paired ECG + Echo data for fusion pipeline.

ECG:  WFDB format → (1000, 12) normalized signals
Echo: DICOM format → (3, 32, 224, 224) video tensors
"""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
import pydicom
import cv2
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ECG_ROOT = os.path.expanduser(
    "~/fusion_project/ecg_data/MIMIC IV ECGs/"
    "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files"
)
ECHO_ROOT = "/proj/faisalflab/projects/mimic-echo-data"


def ecg_path(subject_id: int, study_id: int) -> str:
    """Return base path (without extension) for a WFDB ECG record."""
    sid = str(subject_id)
    first4 = sid[:4]
    return os.path.join(
        ECG_ROOT,
        f"p{first4}",
        f"p{sid}",
        f"s{study_id}",
        str(study_id),
    )


def echo_paths(subject_id: int, study_id: int) -> list[str]:
    """Return sorted list of DICOM file paths for an echo study."""
    sid = str(subject_id)
    first2 = sid[:2]
    pattern = os.path.join(
        ECHO_ROOT,
        f"p{first2}",
        f"p{sid}",
        f"s{study_id}",
        f"{study_id}_*.dcm",
    )
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# ECG loading
# ---------------------------------------------------------------------------

def load_ecg(
    subject_id: int,
    study_id: int,
    stats_path: str = "preprocessing_stats_code15.npz",
    target_length: int = 1000,
) -> np.ndarray:
    """
    Load a 12-lead ECG from WFDB and normalize it.

    Returns:
        np.ndarray of shape (target_length, 12), float32
    """
    record_path = ecg_path(subject_id, study_id)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)  # (n_samples, 12)

    # Resample / pad / truncate to target_length
    n = signal.shape[0]
    if n > target_length:
        signal = signal[:target_length]
    elif n < target_length:
        pad = np.zeros((target_length - n, 12), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=0)

    # Normalize using precomputed stats
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)   # (12,)
    std = stats["std"].astype(np.float32)      # (12,)
    std = np.where(std == 0, 1.0, std)
    signal = (signal - mean) / std

    return signal


# ---------------------------------------------------------------------------
# Echo loading
# ---------------------------------------------------------------------------

def _read_dicom_frame(dcm_path: str, size: int = 224) -> np.ndarray:
    """Read a single DICOM file and return an RGB frame (size, size, 3)."""
    ds = pydicom.dcmread(dcm_path)
    pixel = ds.pixel_array  # (H, W) or (H, W, 3)

    if pixel.ndim == 2:
        pixel = np.stack([pixel] * 3, axis=-1)
    elif pixel.shape[-1] == 1:
        pixel = np.concatenate([pixel] * 3, axis=-1)

    # Ensure uint8 range
    if pixel.dtype != np.uint8:
        pixel = ((pixel - pixel.min()) / (pixel.max() - pixel.min() + 1e-8) * 255).astype(np.uint8)

    frame = cv2.resize(pixel, (size, size))
    return frame  # (224, 224, 3)


def load_echo(
    subject_id: int,
    study_id: int,
    n_frames: int = 32,
    size: int = 224,
) -> np.ndarray:
    """
    Load echo DICOM files and return a video tensor.

    Returns:
        np.ndarray of shape (3, n_frames, size, size), float32, values in [0,1]
    """
    paths = echo_paths(subject_id, study_id)
    if not paths:
        raise FileNotFoundError(
            f"No DICOM files found for subject={subject_id}, study={study_id}"
        )

    frames = []
    for p in paths:
        try:
            frames.append(_read_dicom_frame(p, size))
        except Exception:
            continue

    if len(frames) == 0:
        raise RuntimeError(
            f"Could not read any DICOM frames for subject={subject_id}, study={study_id}"
        )

    # Sample or pad to exactly n_frames
    if len(frames) >= n_frames:
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat last frame to pad
        while len(frames) < n_frames:
            frames.append(frames[-1])

    video = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    video = video.transpose(3, 0, 1, 2)  # (3, T, H, W)
    return video


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """
    Dataset that yields (ecg_signal, echo_video, age_label) tuples.

    Parameters:
        df: DataFrame with columns subject_id, ecg_study_id, echo_study_id,
            age_at_ecg (used as label).
        stats_path: Path to ECG normalization stats .npz file.
        use_cache: If True, cache loaded tensors in memory.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stats_path: str = "preprocessing_stats_code15.npz",
        use_cache: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.stats_path = stats_path
        self.use_cache = use_cache
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        subject_id = int(row["subject_id"])
        ecg_study_id = int(row["ecg_study_id"])
        echo_study_id = int(row["echo_study_id"])
        age = float(row["age_at_ecg"])

        ecg = load_ecg(subject_id, ecg_study_id, self.stats_path)
        echo = load_echo(subject_id, echo_study_id)

        ecg_tensor = torch.from_numpy(ecg)        # (1000, 12)
        echo_tensor = torch.from_numpy(echo)       # (3, 32, 224, 224)
        age_tensor = torch.tensor(age, dtype=torch.float32)

        sample = (ecg_tensor, echo_tensor, age_tensor)

        if self.use_cache:
            self._cache[idx] = sample

        return sample


# ---------------------------------------------------------------------------
# Precomputed-feature Dataset (for faster fusion training)
# ---------------------------------------------------------------------------

class PrecomputedFeatureDataset(Dataset):
    """
    Dataset over pre-extracted feature vectors (no raw data loading).

    Parameters:
        ecg_features: np.ndarray of shape (N, 128)
        echo_features: np.ndarray of shape (N, 1024)
        labels: np.ndarray of shape (N,)
    """

    def __init__(
        self,
        ecg_features: np.ndarray,
        echo_features: np.ndarray,
        labels: np.ndarray,
    ):
        self.ecg_features = torch.from_numpy(ecg_features.astype(np.float32))
        self.echo_features = torch.from_numpy(echo_features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.ecg_features[idx], self.echo_features[idx], self.labels[idx]
