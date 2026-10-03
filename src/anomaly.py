"""Anomaly detector = PyTorch LSTM autoencoder + level-deviation score.

The autoencoder (trained on normal windows) catches *pattern* anomalies — noisy/erratic
sensors, broken correlations — via reconstruction error. Reconstruction models are known
to reproduce slow level drifts quite well, so a robust level-deviation score (|mean z| over
the last 6h) covers drift. Each per-sensor score is normalised by its 99th percentile on
normal validation data; the combined per-sensor score is the max of the two, and the
window score is the max over sensors. Score > 1 means "beyond 99% of normal behaviour".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.knowledge import SENSORS

WIN = 24
RECENT = 6
CLIP = 8.0


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat: int = len(SENSORS), hidden: int = 48, latent: int = 16):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):                     # x: (B, T, F)
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])
        rep = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)


def windows_ending_at(z: pd.DataFrame, ends) -> tuple[np.ndarray, np.ndarray]:
    """(windows (N, WIN, F), ok mask) for 24h windows ending at each timestamp; ok=False if gaps."""
    arr = np.clip(z.to_numpy(np.float32), -CLIP, CLIP)
    pos = z.index.get_indexer(pd.DatetimeIndex(ends))
    out = np.zeros((len(pos), WIN, arr.shape[1]), np.float32)
    ok = np.zeros(len(pos), bool)
    for i, p in enumerate(pos):
        if p >= WIN - 1:
            w = arr[p - WIN + 1: p + 1]
            if not np.isnan(w).any():
                out[i], ok[i] = w, True
    return out, ok


def raw_scores(model: nn.Module, windows: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Per-sensor (recon_error_recent, level_deviation) — both (N, F)."""
    model.eval()
    rec = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            x = torch.from_numpy(windows[i: i + batch])
            rec.append(((model(x) - x) ** 2)[:, -RECENT:, :].mean(dim=1).numpy())
    rec = np.concatenate(rec) if rec else np.zeros((0, len(SENSORS)), np.float32)
    level = np.abs(windows[:, -RECENT:, :].mean(axis=1))
    return rec, level


class AnomalyDetector:
    def __init__(self, model: LSTMAutoencoder, meta: dict):
        self.model, self.meta = model, meta
        self.rec_q = np.array(meta["recon_q99"])
        self.lvl_q = np.array(meta["level_q99"])

    @classmethod
    def load(cls, model_dir=config.MODEL_DIR) -> "AnomalyDetector":
        meta = json.loads((model_dir / "anomaly_meta.json").read_text())
        model = LSTMAutoencoder()
        model.load_state_dict(torch.load(model_dir / "anomaly_lstm_ae.pt", map_location="cpu"))
        return cls(model, meta)

    def score_windows(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        rec, lvl = raw_scores(self.model, windows)
        rec_n, lvl_n = rec / self.rec_q, lvl / self.lvl_q
        per_sensor = np.maximum(rec_n, lvl_n)
        return {"score": per_sensor.max(axis=1), "per_sensor": per_sensor, "recon": rec_n, "level": lvl_n}
