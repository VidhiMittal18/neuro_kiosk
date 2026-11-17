"""Shared preprocessing and ANN inference for the bundled DenseNet121 models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ANN_WEIGHTS_PATH = MODEL_DIR / "densenet121_pneumonia_ann_state_dict.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"
DEVICE = torch.device("cpu")
CLASS_NAMES = ("Normal", "Pneumonia")
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@dataclass(frozen=True)
class LoadedANN:
    model: nn.Module
    threshold: float
    device: torch.device


def read_metadata() -> dict:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["class_to_index"] != {"NORMAL": 0, "PNEUMONIA": 1}:
        raise ValueError("The bundled models must classify NORMAL as 0 and PNEUMONIA as 1.")
    return metadata


def load_ann(device: torch.device = DEVICE) -> LoadedANN:
    metadata = read_metadata()
    threshold = float(metadata["thresholds_selected_on_validation"]["ann"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The ANN validation threshold must be between 0 and 1.")

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    state = torch.load(ANN_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return LoadedANN(model=model.to(device).eval(), threshold=threshold, device=device)


def predict_ann(model: LoadedANN, image: Image.Image) -> dict:
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(model.device)
    with torch.inference_mode():
        logit = model.model(tensor).reshape(-1)[0]
        pneumonia_score = float(torch.sigmoid(logit).item())

    is_pneumonia = pneumonia_score >= model.threshold
    return {
        "prediction": CLASS_NAMES[1] if is_pneumonia else CLASS_NAMES[0],
        "confidence": pneumonia_score if is_pneumonia else 1.0 - pneumonia_score,
        "pneumonia_score": pneumonia_score,
        "threshold": model.threshold,
        "source": "Cloud-DenseNet121-ANN",
    }
