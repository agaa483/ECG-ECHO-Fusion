"""
train.py — Train the fusion head on precomputed ECG + Echo features.

Workflow:
  1. Load paireddata.csv
  2. Split 80/20 by patient (subject_id)
  3. Extract & cache features from both pretrained models
  4. Train a small MLP fusion head
  5. Log MAE, RMSE, R² each epoch
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import FusionDataset, PrecomputedFeatureDataset
from models import ECGFeatureExtractor, EchoFeatureExtractor, FusionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction / caching
# ---------------------------------------------------------------------------

def extract_and_cache_features(
    df: pd.DataFrame,
    cache_dir: str,
    stats_path: str,
    ecg_model_path: str,
    echo_model_path: str,
    ecg_batch_size: int = 64,
    echo_batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from both pretrained models for every row in df.
    Caches results to disk so subsequent runs skip extraction.

    Returns:
        ecg_feats  (N, 128)
        echo_feats (N, 1024)
        labels     (N,)
    """
    os.makedirs(cache_dir, exist_ok=True)
    ecg_cache = os.path.join(cache_dir, "ecg_features.npy")
    echo_cache = os.path.join(cache_dir, "echo_features.npy")
    label_cache = os.path.join(cache_dir, "labels.npy")
    index_cache = os.path.join(cache_dir, "index.npy")

    # Check if cache exists and matches current dataframe
    if all(os.path.exists(p) for p in [ecg_cache, echo_cache, label_cache, index_cache]):
        cached_idx = np.load(index_cache)
        if np.array_equal(cached_idx, df.index.values):
            logger.info("Loading cached features from %s", cache_dir)
            return np.load(ecg_cache), np.load(echo_cache), np.load(label_cache)

    logger.info("Extracting features (this may take a while)...")

    # ---- ECG features (TensorFlow) ----
    logger.info("Loading ECG feature extractor...")
    ecg_extractor = ECGFeatureExtractor(model_path=ecg_model_path)

    dataset = FusionDataset(df, stats_path=stats_path)

    ecg_feats_list = []
    echo_inputs = []
    labels = []

    for i in range(len(dataset)):
        try:
            ecg_signal, echo_video, age = dataset[i]
            ecg_feats_list.append(ecg_signal.numpy())
            echo_inputs.append(echo_video.numpy())
            labels.append(age.item())
        except Exception as e:
            logger.warning("Skipping index %d: %s", i, e)
            ecg_feats_list.append(None)
            echo_inputs.append(None)
            labels.append(None)

    # Filter out failed samples
    valid = [i for i in range(len(labels)) if labels[i] is not None]
    logger.info("Successfully loaded %d / %d samples", len(valid), len(dataset))

    ecg_batch = np.stack([ecg_feats_list[i] for i in valid])   # (N, 1000, 12)
    labels_arr = np.array([labels[i] for i in valid], dtype=np.float32)

    # Extract ECG features in batches
    ecg_all = []
    for start in range(0, len(ecg_batch), ecg_batch_size):
        batch = ecg_batch[start : start + ecg_batch_size]
        ecg_all.append(ecg_extractor.extract(batch))
    ecg_feats = np.concatenate(ecg_all, axis=0)  # (N, 128)
    logger.info("ECG features extracted: %s", ecg_feats.shape)

    # Free TF memory
    del ecg_extractor
    import gc; gc.collect()

    # ---- Echo features (PyTorch) ----
    logger.info("Loading Echo feature extractor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    echo_extractor = EchoFeatureExtractor(checkpoint_path=echo_model_path, device=device)

    echo_all = []
    for start in range(0, len(valid), echo_batch_size):
        batch_indices = valid[start : start + echo_batch_size]
        batch = torch.from_numpy(
            np.stack([echo_inputs[i] for i in batch_indices])
        )
        feats = echo_extractor.extract(batch)
        echo_all.append(feats)
    echo_feats = np.concatenate(echo_all, axis=0)  # (N, 1024)
    logger.info("Echo features extracted: %s", echo_feats.shape)

    del echo_extractor
    torch.cuda.empty_cache()

    # Save cache
    np.save(ecg_cache, ecg_feats)
    np.save(echo_cache, echo_feats)
    np.save(label_cache, labels_arr)
    np.save(index_cache, df.index.values)
    logger.info("Features cached to %s", cache_dir)

    return ecg_feats, echo_feats, labels_arr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: FusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for ecg_f, echo_f, label in loader:
        ecg_f = ecg_f.to(device)
        echo_f = echo_f.to(device)
        label = label.to(device)

        pred = model(ecg_f, echo_f)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label)
        n += len(label)

    return total_loss / n


@torch.no_grad()
def evaluate(
    model: FusionModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    preds, targets = [], []
    for ecg_f, echo_f, label in loader:
        ecg_f = ecg_f.to(device)
        echo_f = echo_f.to(device)
        pred = model(ecg_f, echo_f)
        preds.append(pred.cpu().numpy())
        targets.append(label.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    return {"mae": mae, "rmse": rmse, "r2": r2, "preds": preds, "targets": targets}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load paired data
    df = pd.read_csv(args.data_csv)
    logger.info("Loaded %d paired records", len(df))

    # Patient-level 80/20 split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df["subject_id"].values
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Split: %d train samples (%d patients), %d val samples (%d patients)",
        len(df_train), df_train["subject_id"].nunique(),
        len(df_val), df_val["subject_id"].nunique(),
    )

    # Extract features
    logger.info("--- Extracting TRAIN features ---")
    ecg_train, echo_train, y_train = extract_and_cache_features(
        df_train,
        cache_dir=os.path.join(args.output_dir, "cache_train"),
        stats_path=args.stats_path,
        ecg_model_path=args.ecg_model,
        echo_model_path=args.echo_model,
    )

    logger.info("--- Extracting VAL features ---")
    ecg_val, echo_val, y_val = extract_and_cache_features(
        df_val,
        cache_dir=os.path.join(args.output_dir, "cache_val"),
        stats_path=args.stats_path,
        ecg_model_path=args.ecg_model,
        echo_model_path=args.echo_model,
    )

    # Datasets & loaders
    train_ds = PrecomputedFeatureDataset(ecg_train, echo_train, y_train)
    val_ds = PrecomputedFeatureDataset(ecg_val, echo_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    logger.info("Training fusion head on %s", device)
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["mae"])

        logger.info(
            "Epoch %3d | train_loss %.4f | val_MAE %.3f | val_RMSE %.3f | val_R² %.4f",
            epoch, train_loss,
            val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"],
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            ckpt_path = os.path.join(args.output_dir, "best_fusion_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  -> Saved best model (MAE %.3f)", best_mae)

    logger.info("Training complete. Best val MAE: %.3f", best_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG+Echo fusion head")
    parser.add_argument("--data-csv", default="paireddata.csv")
    parser.add_argument("--stats-path", default="preprocessing_stats_code15.npz")
    parser.add_argument("--ecg-model", default="best_model_code15_artifact_aware.h5")
    parser.add_argument("--echo-model", default="EchoFM.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
