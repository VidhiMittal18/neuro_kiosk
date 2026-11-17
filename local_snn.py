"""Inference helpers for the bundled DenseNet121 SpikingJelly model."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from spikingjelly.activation_based import functional

from team_model import DEVICE, PREPROCESS, read_metadata


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "densenet121_pneumonia_snn.pt"


@dataclass(frozen=True)
class LoadedSNN:
    model: nn.Module
    threshold: float
    timesteps: int
    device: torch.device


def load_model(device: torch.device = DEVICE) -> LoadedSNN:
    metadata = read_metadata()
    timesteps = int(metadata["models"]["best_converted_snn"]["timesteps"])
    if timesteps not in (4, 8, 16):
        raise ValueError(f"Unsupported SNN timestep count in metadata: {timesteps}.")
    threshold = float(metadata["thresholds_selected_on_validation"]["snn"][str(timesteps)])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The SNN validation threshold must be between 0 and 1.")

    # This trusted bundled checkpoint is a serialized FX GraphModule, not a state dict.
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError("The saved SNN checkpoint is not a torch module.")
    functional.reset_net(model)
    return LoadedSNN(model=model.to(device).eval(), threshold=threshold, timesteps=timesteps, device=device)


def predict(model: LoadedSNN, image: Image.Image) -> dict:
    started = time.perf_counter()
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(model.device)
    functional.reset_net(model.model)
    try:
        total_logits = None
        with torch.inference_mode():
            for _ in range(model.timesteps):
                logits_t = model.model(tensor)
                total_logits = logits_t if total_logits is None else total_logits + logits_t
            pneumonia_score = float(
                torch.sigmoid((total_logits / model.timesteps).reshape(-1)[0]).item()
            )
    finally:
        functional.reset_net(model.model)

    is_pneumonia = pneumonia_score >= model.threshold
    return {
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "confidence": pneumonia_score if is_pneumonia else 1.0 - pneumonia_score,
        "pneumonia_score": pneumonia_score,
        "threshold": model.threshold,
        "time_steps": model.timesteps,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "source": "Edge-SNN (local CPU)",
    }
