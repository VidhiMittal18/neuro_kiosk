"""Explainability service for the converted DenseNet121 SpikingJelly SNN.

Place this file at neuro_kiosk/explainability/xai_service.py.
The service loads the trusted FX GraphModule checkpoint once and exposes
per-image explanation generation for the Neuro Kiosk API.

Methods:
  grad2d, grad3d, ig2d, ig3d, sam, inverted_sam, cam

Pixel deletion is an evaluation experiment and is intentionally exposed as
an offline evaluation helper rather than a routine single-click heatmap.
"""
from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from spikingjelly.activation_based import functional


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_SAM_LAYER = "ltb_spiking_0.if_node"
SUPPORTED_TIMESTEPS = (8, 16)
METHODS = {
    "grad2d", "grad3d", "ig2d", "ig3d",
    "sam", "inverted_sam", "cam", "pixel_deletion",
}


class DenseNetSNNXAI:
    """State-safe XAI wrapper for the saved binary DenseNet121 SNN."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        device: str | torch.device = "cpu",
        sam_layer_name: str = DEFAULT_SAM_LAYER,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        self.device = torch.device(device)
        self._lock = threading.RLock()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"SNN checkpoint not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        threshold_config = self.metadata.get("thresholds_selected_on_validation", {})
        snn_thresholds = threshold_config.get("snn", {})
        self.thresholds = {
            8: float(snn_thresholds.get("8", 0.9987940788269043)),
            16: float(snn_thresholds.get("16", 0.9999173879623413)),
        }

        # The saved model is a trusted serialized FX GraphModule. Do not load
        # arbitrary checkpoints from untrusted sources with weights_only=False.
        self.model = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(self.model, nn.Module):
            raise TypeError("The saved SNN checkpoint is not a torch.nn.Module.")
        self.model = self.model.to(self.device).eval()
        functional.reset_net(self.model)

        self.sam_layer_name = sam_layer_name
        try:
            self.sam_layer = self.model.get_submodule(self.sam_layer_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not find final spiking feature layer '{self.sam_layer_name}'."
            ) from exc

        if "IFNode" not in type(self.sam_layer).__name__:
            raise TypeError(
                f"Expected an IF spiking module at {self.sam_layer_name}, "
                f"found {type(self.sam_layer).__name__}."
            )

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply the exact DenseNet121 preprocessing used for evaluation."""
        return self.preprocess(image.convert("L")).unsqueeze(0).to(self.device)

    def _reset(self) -> None:
        functional.reset_net(self.model)
        self.model.zero_grad(set_to_none=True)

    def _predict_tensor(self, x: torch.Tensor, timesteps: int) -> dict[str, Any]:
        if timesteps not in SUPPORTED_TIMESTEPS:
            raise ValueError(f"Supported XAI timesteps are {SUPPORTED_TIMESTEPS}.")
        self._reset()
        try:
            step_logits = []
            with torch.no_grad():
                for _ in range(timesteps):
                    out = self.model(x)
                    if not torch.is_tensor(out):
                        raise TypeError("SNN returned a non-tensor output.")
                    out = out.reshape(x.shape[0], -1)
                    if out.shape[1] != 1:
                        raise RuntimeError(f"Expected one binary logit; got {tuple(out.shape)}")
                    step_logits.append(out[:, 0])
            mean_logit = torch.stack(step_logits, dim=0).mean(dim=0)
            score = float(torch.sigmoid(mean_logit[0]).item())
            threshold = self.thresholds[timesteps]
            pred = int(score >= threshold)
            return {
                "prediction_index": pred,
                "prediction": "PNEUMONIA" if pred == 1 else "NORMAL",
                "pneumonia_score": score,
                "threshold": threshold,
                "mean_logit": float(mean_logit[0].item()),
            }
        finally:
            self._reset()

    @staticmethod
    def _target_sign(target_class: int) -> float:
        if target_class not in (0, 1):
            raise ValueError("target_class must be 0 (NORMAL) or 1 (PNEUMONIA).")
        return 1.0 if target_class == 1 else -1.0

    def _sequence_target(
        self,
        sequence: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        logits = []
        for t in range(sequence.shape[0]):
            out = self.model(sequence[t]).reshape(-1)
            if out.numel() != 1:
                raise RuntimeError(f"Expected one binary logit; got {tuple(out.shape)}")
            logits.append(out[0])
        return self._target_sign(target_class) * torch.stack(logits).sum()

    @staticmethod
    def _reduce_channels(attr: torch.Tensor, mode: str = "abs_sum") -> torch.Tensor:
        if attr.ndim == 4:       # [T,C,H,W]
            channel_dim = 1
        elif attr.ndim == 3:     # [C,H,W]
            channel_dim = 0
        else:
            raise ValueError(f"Expected [T,C,H,W] or [C,H,W], got {tuple(attr.shape)}")
        if mode == "abs_sum":
            return attr.abs().sum(dim=channel_dim)
        if mode == "abs_mean":
            return attr.abs().mean(dim=channel_dim)
        if mode == "sum":
            return attr.sum(dim=channel_dim)
        if mode == "max_abs":
            return attr.abs().max(dim=channel_dim).values
        raise ValueError(f"Unknown channel reduction: {mode}")

    @staticmethod
    def _normalize_map(heatmap: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
        h = heatmap.detach().float().cpu()
        h = h - h.min()
        h = h / (h.max() + eps)
        return h.numpy()

    @staticmethod
    def _upsample_maps(maps: torch.Tensor, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
        if maps.ndim == 2:
            maps = maps.unsqueeze(0)
        if maps.ndim != 3:
            raise ValueError(f"Expected [T,H,W] or [H,W], got {tuple(maps.shape)}")
        return F.interpolate(
            maps.unsqueeze(1), size=size, mode="bilinear", align_corners=False
        ).squeeze(1)

    def _baseline(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "zero":
            return torch.zeros_like(x)
        values = {"black": 0.0, "gray": 0.5, "white": 1.0}
        if mode not in values:
            raise ValueError(f"Unknown baseline mode: {mode}")
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        raw = torch.full_like(x, values[mode])
        return (raw - mean) / std

    def _grad3d(self, x: torch.Tensor, timesteps: int, target_class: int) -> torch.Tensor:
        sequence = x.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1).detach().clone().requires_grad_(True)
        self._reset()
        try:
            with torch.enable_grad():
                target = self._sequence_target(sequence, target_class)
                gradient = torch.autograd.grad(target, sequence, retain_graph=False)[0]
            return self._reduce_channels(gradient[:, 0], "abs_sum").detach()
        finally:
            self._reset()

    def _ig2d(self, x: torch.Tensor, timesteps: int, target_class: int, n_steps: int = 50) -> torch.Tensor:
        baseline = self._baseline(x, "black")
        total_grad = torch.zeros_like(x)
        for i in range(1, n_steps + 1):
            alpha = float(i) / n_steps
            x_alpha = (baseline + alpha * (x - baseline)).detach().clone().requires_grad_(True)
            sequence = x_alpha.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
            self._reset()
            try:
                with torch.enable_grad():
                    target = self._sequence_target(sequence, target_class)
                    gradient = torch.autograd.grad(target, x_alpha)[0]
                total_grad += gradient.detach()
            finally:
                self._reset()
        integrated = (x - baseline) * (total_grad / n_steps)
        return self._reduce_channels(integrated[0], "abs_sum").detach()

    def _ig3d(self, x: torch.Tensor, timesteps: int, target_class: int, n_steps: int = 50) -> torch.Tensor:
        full_sequence = x.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
        baseline_image = self._baseline(x, "zero")
        baseline_sequence = baseline_image.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
        total_grad = torch.zeros_like(full_sequence)
        for i in range(1, n_steps + 1):
            k = max(1, int(np.ceil(timesteps * i / n_steps)))
            indices = torch.linspace(0, timesteps - 1, steps=k, device=x.device).round().long().unique()
            sequence = baseline_sequence.clone()
            sequence[indices] = full_sequence[indices]
            sequence = sequence.detach().clone().requires_grad_(True)
            self._reset()
            try:
                with torch.enable_grad():
                    target = self._sequence_target(sequence, target_class)
                    gradient = torch.autograd.grad(target, sequence)[0]
                total_grad += gradient.detach()
            finally:
                self._reset()
        integrated = (full_sequence - baseline_sequence) * (total_grad / n_steps)
        return self._reduce_channels(integrated[:, 0], "abs_sum").detach()

    @staticmethod
    def _sam_from_spikes(feature_spikes: torch.Tensor, gamma: float = 0.5, trail: int = 5) -> torch.Tensor:
        # feature_spikes: [T,1,C,h,w], matching the mentor's implementation.
        feat = feature_spikes[:, 0].float()
        timesteps, channels, height, width = feat.shape
        maps = []
        for t in range(timesteps):
            start = max(0, t - trail + 1)
            end = t + 1
            taus = torch.arange(start, end, device=feat.device)
            weights = torch.exp(-gamma * (t - taus).float()).view(-1, 1, 1, 1)
            ncs = (feat[start:end] * weights).sum(dim=0)
            maps.append((ncs * feat[t]).sum(dim=0))
        return torch.stack(maps, dim=0)

    def _record_sam(self, x: torch.Tensor, timesteps: int) -> torch.Tensor:
        records: list[torch.Tensor] = []
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("SAM layer returned a non-tensor.")
            records.append(output.detach().clone())
        handle = self.sam_layer.register_forward_hook(hook)
        self._reset()
        try:
            with torch.no_grad():
                for _ in range(timesteps):
                    self.model(x)
        finally:
            handle.remove()
            self._reset()
        if len(records) != timesteps:
            raise RuntimeError(f"Expected {timesteps} SAM records; got {len(records)}")
        return torch.stack(records, dim=0)

    def _cam(self, x: torch.Tensor, timesteps: int, target_class: int, global_time_weight: bool = True) -> torch.Tensor:
        activations: list[torch.Tensor] = []
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError("SNN-CAM activation does not retain gradients.")
            output.retain_grad()
            activations.append(output)
        handle = self.sam_layer.register_forward_hook(hook)
        self._reset()
        try:
            with torch.enable_grad():
                logits = []
                for _ in range(timesteps):
                    logits.append(self.model(x).reshape(-1)[0])
                target = self._target_sign(target_class) * torch.stack(logits).sum()
                # Activations are populated by the forward hooks above.
                # Request their gradients with respect to the scalar target.
                acts = list(activations)
                gradients = torch.autograd.grad(target, acts, retain_graph=False, allow_unused=False)
            act_tensor = torch.stack([a.detach()[0] for a in acts], dim=0)  # [T,C,h,w]
            grad_tensor = torch.stack([g.detach()[0] for g in gradients], dim=0)
            if global_time_weight:
                weights = grad_tensor.mean(dim=(0, 2, 3))
                cam_small = (act_tensor * weights.view(1, -1, 1, 1)).sum(dim=1)
            else:
                weights_t = grad_tensor.mean(dim=(2, 3))
                cam_small = (act_tensor * weights_t[:, :, None, None]).sum(dim=1)
            cam_small = F.relu(cam_small)
            return self._upsample_maps(cam_small)
        finally:
            handle.remove()
            self._reset()

    @staticmethod
    def _pil_png_data_uri(fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/png;base64," + encoded

    def _render_spatial(self, image: Image.Image, heatmap: torch.Tensor, title: str) -> tuple[str, str]:
        raw = np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
        heat = self._normalize_map(heatmap)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(raw, cmap="gray")
        axes[0].set_title("Original X-ray")
        axes[1].imshow(raw, cmap="gray")
        axes[1].imshow(heat, cmap="jet", alpha=0.45)
        axes[1].set_title(title + " overlay")
        for ax in axes:
            ax.axis("off")
        return self._pil_png_data_uri(fig), self._render_heat_only(heat, title)

    def _render_heat_only(self, heat: np.ndarray, title: str) -> str:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        return self._pil_png_data_uri(fig)

    def _render_temporal(self, image: Image.Image, maps: torch.Tensor, title: str) -> str:
        raw = np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
        global_max = float(maps.max().item())
        if global_max <= 0:
            global_max = 1.0
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for t, ax in enumerate(axes.flat):
            current = maps[t].detach().cpu().numpy()
            current = current - current.min()
            normalized = current / (current.max() + 1e-8)
            ax.imshow(raw, cmap="gray")
            ax.imshow(normalized, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"Timestep {t + 1}")
            ax.axis("off")
        fig.suptitle(title)
        return self._pil_png_data_uri(fig)

    def explain(
        self,
        image: Image.Image,
        method: str,
        timesteps: int = 16,
        target_class: int | None = None,
        ig_steps: int = 50,
    ) -> dict[str, Any]:
        method = method.lower().strip()
        if method not in METHODS - {"pixel_deletion"}:
            raise ValueError(f"Unsupported explanation method: {method}")
        if timesteps not in SUPPORTED_TIMESTEPS:
            raise ValueError(f"Supported timesteps: {SUPPORTED_TIMESTEPS}")
        if ig_steps < 2 or ig_steps > 50:
            raise ValueError("ig_steps must be between 2 and 50")

        with self._lock:
            x = self.preprocess_image(image)
            prediction = self._predict_tensor(x, timesteps)
            pred_class = int(prediction["prediction_index"])
            explanation_class = pred_class if target_class is None else int(target_class)

            if method == "grad3d":
                temporal = self._grad3d(x, timesteps, explanation_class)
                display_map = temporal.mean(dim=0)
            elif method == "grad2d":
                temporal = self._grad3d(x, timesteps, explanation_class)
                display_map = temporal.sum(dim=0)
            elif method == "ig2d":
                display_map = self._ig2d(x, timesteps, explanation_class, ig_steps)
                temporal = None
            elif method == "ig3d":
                temporal = self._ig3d(x, timesteps, explanation_class, ig_steps)
                display_map = temporal.mean(dim=0)
            elif method in ("sam", "inverted_sam"):
                spikes = self._record_sam(x, timesteps)
                temporal = self._upsample_maps(self._sam_from_spikes(spikes))
                if method == "inverted_sam":
                    temporal = -temporal
                display_map = temporal.mean(dim=0)
            elif method == "cam":
                temporal = self._cam(x, timesteps, explanation_class, global_time_weight=True)
                display_map = temporal.mean(dim=0)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # For inverted SAM, normalize the signed map for display only.
            overlay_uri, heat_uri = self._render_spatial(
                image, display_map, method.upper().replace("_", " ")
            )
            response: dict[str, Any] = {
                **prediction,
                "method": method,
                "target_class": "PNEUMONIA" if explanation_class == 1 else "NORMAL",
                "timesteps": timesteps,
                "heatmap_data_url": heat_uri,
                "overlay_data_url": overlay_uri,
                "note": "Attribution visualisation; not a verified lesion localisation or clinical explanation.",
            }
            if temporal is not None:
                response["temporal_data_url"] = self._render_temporal(
                    image, temporal,
                    method.upper().replace("_", " ") + " temporal maps",
                )
            return response

    def health(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "model": "DenseNet121 SpikingJelly SNN",
            "spiking_layer": self.sam_layer_name,
            "supported_timesteps": list(SUPPORTED_TIMESTEPS),
            "methods": sorted(METHODS - {"pixel_deletion"}),
        }
