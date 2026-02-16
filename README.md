# Multimodal ECG-Echo Cardiac Age Prediction

A deep learning pipeline that fuses ECG and echocardiogram data to predict cardiac age using intermediate fusion.

## Overview

This project combines two pretrained models:
- **ECG Model**: CNN-BiLSTM trained on CODE-15 dataset (92% accuracy) → 128-dimensional embeddings
- **Echo Model**: EchoFM Vision Transformer trained on 290K echo clips (94% accuracy) → 1024-dimensional embeddings

The fusion pipeline concatenates these embeddings (1,152 dimensions) and trains a lightweight fusion head to predict cardiac age.

## Architecture
```
ECG Signal (1000, 12) → CNN-BiLSTM → GlobalAvgPool → 128-dim features ─┐
                                                                        ├─→ Concat → Fusion MLP → Predicted Age
Echo Video (32, 224, 224) → EchoFM ViT → Encoder → 1024-dim features ──┘
```

## Files

- `data_loader.py` — Data loading utilities for WFDB (ECG) and DICOM (Echo) formats
- `models.py` — ECG/Echo feature extractors and fusion model architecture
- `train.py` — Training pipeline with train/val split
- `evaluate.py` — Model evaluation and metrics (MAE, RMSE, R²)

## Data

Trained on MIMIC-IV paired ECG-Echo dataset with 10,000+ patient records.

- ECG: 12-lead, 10-second recordings (WFDB format)
- Echo: Multi-frame echocardiogram videos (DICOM format)

## Models (not included due to size)

- `best_model_code15_artifact_aware.h5` — Pretrained ECG CNN-BiLSTM (~1GB)
- `EchoFM.pth` — Pretrained Echo Vision Transformer (~1.2GB)

## Environment
```bash
pip install tensorflow torch numpy pandas wfdb pydicom opencv-python
```

## Usage
```bash
python train.py
python evaluate.py
```

## Infrastructure

Deployed on UNC Longleaf HPC cluster using SLURM job scheduling and NVIDIA L40S GPUs.
