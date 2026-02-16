"""
evaluate.py — Evaluate the trained fusion model and save predictions to CSV.

Usage:
    python evaluate.py --data-csv paireddata.csv --fusion-model outputs/best_fusion_model.pt
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import PrecomputedFeatureDataset
from models import FusionModel
from train import extract_and_cache_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(args):
    # Load data and reproduce the same split used in training
    df = pd.read_csv(args.data_csv)
    logger.info("Loaded %d paired records", len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df["subject_id"].values
    _, val_idx = next(splitter.split(df, groups=groups))
    df_val = df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Evaluation set: %d samples (%d patients)",
        len(df_val), df_val["subject_id"].nunique(),
    )

    # Load or extract cached features
    ecg_feats, echo_feats, labels = extract_and_cache_features(
        df_val,
        cache_dir=os.path.join(args.cache_dir, "cache_val"),
        stats_path=args.stats_path,
        ecg_model_path=args.ecg_model,
        echo_model_path=args.echo_model,
    )

    dataset = PrecomputedFeatureDataset(ecg_feats, echo_feats, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load fusion model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    state = torch.load(args.fusion_model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Run inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ecg_f, echo_f, label in loader:
            ecg_f = ecg_f.to(device)
            echo_f = echo_f.to(device)
            pred = model(ecg_f, echo_f)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Metrics
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info("  MAE  : %.3f years", mae)
    logger.info("  RMSE : %.3f years", rmse)
    logger.info("  R²   : %.4f", r2)
    logger.info("  N    : %d", len(preds))
    logger.info("=" * 50)

    # Save predictions CSV
    results_df = df_val.iloc[: len(preds)].copy()
    results_df["predicted_age"] = preds
    results_df["true_age"] = targets
    results_df["error"] = preds - targets
    results_df["abs_error"] = np.abs(preds - targets)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)

    # Save summary metrics
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"MAE:  {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"N:    {len(preds)}\n")
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fusion model")
    parser.add_argument("--data-csv", default="paireddata.csv")
    parser.add_argument("--fusion-model", default="outputs/best_fusion_model.pt")
    parser.add_argument("--stats-path", default="preprocessing_stats_code15.npz")
    parser.add_argument("--ecg-model", default="best_model_code15_artifact_aware.h5")
    parser.add_argument("--echo-model", default="EchoFM.pth")
    parser.add_argument("--cache-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(args)
