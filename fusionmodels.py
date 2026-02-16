"""
models.py — Feature extractors and FusionModel for ECG + Echo cardiac age prediction.

ECG feature extractor:  TensorFlow CNN-BiLSTM → GlobalAveragePooling1D → 128-dim
Echo feature extractor: PyTorch MAE ViT-Large → CLS token → 1024-dim
Fusion head:            MLP on concatenated 1152-dim features → age prediction
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf


# ---------------------------------------------------------------------------
# ECG Feature Extractor (TensorFlow)
# ---------------------------------------------------------------------------

class ECGFeatureExtractor:
    """
    Wraps the pretrained CNN-BiLSTM ECG model and exposes a method
    to extract 128-dim features from the GlobalAveragePooling1D layer.
    """

    def __init__(
        self,
        model_path: str = "best_model_code15_artifact_aware.h5",
        feature_layer: str = "global_average_pooling1d",
    ):
        self.full_model = tf.keras.models.load_model(model_path, compile=False)

        # Build a sub-model that outputs at the pooling layer
        pool_output = self.full_model.get_layer(feature_layer).output
        self.feature_model = tf.keras.Model(
            inputs=self.full_model.input,
            outputs=pool_output,
        )

    def extract(self, ecg_batch: np.ndarray) -> np.ndarray:
        """
        Parameters:
            ecg_batch: np.ndarray of shape (batch, 1000, 12), float32, normalized

        Returns:
            np.ndarray of shape (batch, 128)
        """
        return self.feature_model.predict(ecg_batch, verbose=0)

    def extract_single(self, ecg: np.ndarray) -> np.ndarray:
        """Extract features for a single ECG signal (1000, 12)."""
        return self.extract(ecg[np.newaxis])[0]


# ---------------------------------------------------------------------------
# Echo Feature Extractor (PyTorch)
# ---------------------------------------------------------------------------

class EchoFeatureExtractor(nn.Module):
    """
    Wraps EchoFM (MAE ViT-Large) and extracts the encoder CLS token
    as a 1024-dim feature vector.
    """

    def __init__(
        self,
        checkpoint_path: str = "EchoFM.pth",
        device: str | None = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Import the model constructor from the EchoFM repo
        from EchoFM.models_mae import mae_vit_large_patch16

        self.model = mae_vit_large_patch16()

        # Load pretrained weights
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            video: (batch, 3, 32, 224, 224) float32 tensor in [0, 1]

        Returns:
            features: (batch, 1024) CLS token from the encoder
        """
        video = video.to(self.device)

        # EchoFM encoder forward: returns (N, L, D) where first token is CLS
        latent, _, _ = self.model.forward_encoder(video, mask_ratio=0.0)
        cls_token = latent[:, 0, :]  # (batch, 1024)

        return cls_token

    def extract(self, video: torch.Tensor) -> np.ndarray:
        """Same as forward but returns numpy on CPU."""
        return self.forward(video).cpu().numpy()


# ---------------------------------------------------------------------------
# Fusion Model (PyTorch)
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    Small MLP fusion head that takes concatenated ECG (128-dim) and
    Echo (1024-dim) features and predicts cardiac age.

    Architecture:
        1152 → 256 (ReLU, Dropout) → 64 (ReLU, Dropout) → 1
    """

    def __init__(self, ecg_dim: int = 128, echo_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        in_dim = ecg_dim + echo_dim

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ecg_features: torch.Tensor,
        echo_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
            ecg_features:  (batch, 128)
            echo_features: (batch, 1024)

        Returns:
            predictions: (batch,) — predicted cardiac age
        """
        x = torch.cat([ecg_features, echo_features], dim=1)  # (batch, 1152)
        return self.head(x).squeeze(-1)
