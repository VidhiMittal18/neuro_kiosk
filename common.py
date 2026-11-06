"""
common.py - single source of truth shared by training, the cloud API and the edge kiosk.

Everything that MUST be identical across the three runtimes lives here:
  * image preprocessing / normalisation  (get_transform / preprocess_pil)
  * model architectures + hyper-parameters (CNNBaseline, SpikingCNN, NUM_STEPS, BETA, SLOPE)
  * score -> probability -> label logic (model_scores / predict_probs)
  * SNN activity accounting (spike sparsity + synaptic operations)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import snntorch as snn
from snntorch import surrogate

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"
CNN_PATH = ROOT / "trained_cnn.pt"
SNN_PATH = ROOT / "trained_snn.pt"

CLASS_NAMES = ("Normal", "Pneumonia")      # PneumoniaMNIST: 0 = normal, 1 = pneumonia
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 28
NORM_MEAN, NORM_STD = (0.5,), (0.5,)       # -> pixel range [-1, 1]

NUM_STEPS = 15                             # SNN simulation time-steps (spec: 10-20)
BETA = 0.9                                 # LIF membrane decay
SLOPE = 25.0                               # fast-sigmoid surrogate slope
LAYERS = ("lif1", "lif2", "lif3")


# --------------------------------------------------------------------------- #
# Preprocessing (identical in training, FastAPI endpoint and edge app)
# --------------------------------------------------------------------------- #
def _to_gray_28(img: Image.Image) -> Image.Image:
    """Force 1-channel 28x28. No-op for MedMNIST images; resizes arbitrary uploads."""
    img = img.convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return img


def get_transform() -> transforms.Compose:
    """THE transform. Used by the training DataLoaders AND by preprocess_pil()."""
    return transforms.Compose([
        transforms.Lambda(_to_gray_28),
        transforms.ToTensor(),                              # uint8 [0,255] -> float [0,1]
        transforms.Normalize(NORM_MEAN, NORM_STD),          # -> [-1, 1]
    ])


_TRANSFORM = get_transform()


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """PIL image -> float tensor [1, 1, 28, 28], same pipeline as training."""
    return _TRANSFORM(img).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CNNBaseline(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool2d (x2) -> Linear."""
    is_spiking = False

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 14->7
        )
        self.classifier = nn.Linear(32 * 7 * 7, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


class SpikingCNN(nn.Module):
    """
    Lightweight convolutional SNN (snnTorch Leaky Integrate-and-Fire neurons).

        x --conv1--pool--> [LIF1] --conv2--pool--> [LIF2] --fc--> [LIF3] -> output spikes

    Input coding: *direct encoding*. The (constant) image is injected as an analog current,
    so conv1(x) is computed ONCE and re-used at every time-step. Every layer after conv1
    receives binary spikes, i.e. it is event-driven and counted as synaptic operations (SOPs).

    --- Surrogate gradient -------------------------------------------------------------
    A LIF neuron emits   S[t] = Heaviside(U[t] - threshold).
    dS/dU is 0 everywhere except a Dirac spike at U = threshold, so plain back-prop is dead.
    snntorch.surrogate.fast_sigmoid keeps the exact Heaviside in the FORWARD pass but, in the
    BACKWARD pass, substitutes the smooth derivative
            dS/dU  ~=  1 / (slope * |U - threshold| + 1)^2
    which is largest when the membrane is close to threshold. Gradients then flow through
    time (BPTT) via U[t] = beta * U[t-1] + I[t] - S[t-1] * threshold.
    ------------------------------------------------------------------------------------
    """
    is_spiking = True

    def __init__(self, num_steps: int = NUM_STEPS, beta: float = BETA, slope: float = SLOPE) -> None:
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.conv1 = nn.Conv2d(1, 12, 5)             # 28 -> 24 -> pool 12
        self.conv2 = nn.Conv2d(12, 32, 5)            # 12 -> 8  -> pool 4
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 4 * 4, NUM_CLASSES)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward_record(self, x: torch.Tensor, record: bool = False):
        """Returns (spk_out [T,B,C], mem_out [T,B,C], activity-dict-or-None)."""
        B = x.size(0)
        cur1 = self.pool(self.conv1(x))                                   # constant over time
        h2 = (cur1.shape[-1] - self.conv2.kernel_size[0] + 1) // 2
        mem1 = torch.zeros_like(cur1)
        mem2 = x.new_zeros(B, self.conv2.out_channels, h2, h2)
        mem3 = x.new_zeros(B, NUM_CLASSES)

        spk3_rec, mem3_rec, s1_rec, s2_rec = [], [], [], []
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc(torch.flatten(spk2, 1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            if record:
                s1_rec.append(spk1)
                s2_rec.append(spk2)

        spk_out, mem_out = torch.stack(spk3_rec), torch.stack(mem3_rec)
        act = None
        if record:
            act = {"lif1": torch.stack(s1_rec).detach(),
                   "lif2": torch.stack(s2_rec).detach(),
                   "lif3": spk_out.detach()}
        return spk_out, mem_out, act

    def forward(self, x: torch.Tensor):
        spk_out, mem_out, _ = self.forward_record(x, record=False)
        return spk_out, mem_out


def build_model(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "cnn":
        return CNNBaseline()
    if kind == "snn":
        return SpikingCNN()
    raise ValueError(f"unknown model kind: {kind!r} (expected 'cnn' or 'snn')")


def load_trained(kind: str, path: Path | str, device: str = "cpu") -> nn.Module:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run train_and_benchmark.py first")
    model = build_model(kind)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Scoring (identical decision rule everywhere)
# --------------------------------------------------------------------------- #
def snn_scores(spk_out: torch.Tensor, mem_out: torch.Tensor) -> torch.Tensor:
    """Rate decoding: output spike count per class. The tiny membrane term only breaks ties
    (e.g. both output neurons silent) so the decision is never arbitrary."""
    return spk_out.sum(0) + 1e-3 * mem_out.mean(0)


def model_scores(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model.is_spiking:
        spk_out, mem_out = model(x)
        return snn_scores(spk_out, mem_out)
    return model(x)


def predict_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(model_scores(model, x), dim=1)


# --------------------------------------------------------------------------- #
# SNN accounting: spike sparsity + synaptic operations (SOPs)
# --------------------------------------------------------------------------- #
def _lif1_fanout_map(model: SpikingCNN, spatial: int) -> torch.Tensor:
    """Exact fan-out of every LIF1 neuron into conv2 (border-aware).
    conv_transpose2d of ones counts, for each input position, how many conv2 output positions
    it feeds; multiplied by conv2's output channels."""
    k = model.conv2.kernel_size[0]
    out = spatial - k + 1
    m = F.conv_transpose2d(torch.ones(1, 1, out, out), torch.ones(1, 1, k, k))[0, 0]
    return m * model.conv2.out_channels


def new_activity_acc() -> Dict[str, float]:
    return defaultdict(float)


@torch.no_grad()
def accumulate_activity(model: SpikingCNN, act: Dict[str, torch.Tensor], acc: Dict[str, float]) -> None:
    """SOPs = sum over spiking neurons of (spikes emitted x fan-out of that neuron).
        LIF1 -> conv2 (border-aware fan-out), LIF2 -> fc (fan-out = #classes), LIF3 = read-out (0)."""
    T, B = act["lif1"].shape[:2]
    acc["samples"] += B
    for name in LAYERS:
        s = act[name]
        acc[f"spikes/{name}"] += float(s.sum())
        acc[f"neuron_steps/{name}"] += float(s[0, 0].numel() * T * B)   # neurons x T x B
    fan1 = _lif1_fanout_map(model, act["lif1"].shape[-1]).to(act["lif1"].device)
    acc["sops/lif1"] += float((act["lif1"].sum(dim=(0, 1, 2)) * fan1).sum())
    acc["sops/lif2"] += float(act["lif2"].sum()) * model.fc.out_features
    acc["sops/lif3"] += 0.0


def summarize_activity(acc: Dict[str, float]) -> dict:
    n = max(acc["samples"], 1)
    per_layer, sops_layer = {}, {}
    tot_sp = tot_ns = tot_sops = 0.0
    for name in LAYERS:
        sp, ns = acc[f"spikes/{name}"], acc[f"neuron_steps/{name}"]
        per_layer[name] = sp / ns if ns else 0.0
        sops_layer[name] = acc[f"sops/{name}"] / n
        tot_sp, tot_ns, tot_sops = tot_sp + sp, tot_ns + ns, tot_sops + acc[f"sops/{name}"]
    return {
        "mean_spike_rate_per_neuron_per_step": tot_sp / tot_ns if tot_ns else 0.0,
        "per_layer_spike_rate": per_layer,
        "spikes_per_inference": tot_sp / n,
        "sops_per_inference": tot_sops / n,
        "sops_by_layer_per_inference": sops_layer,
    }


def snn_static_ops(model: SpikingCNN):
    """(encoding_macs, dense_equivalent_synaptic_ops).
    encoding_macs: conv1 on the analog image, executed once per inference (direct encoding).
    dense_equivalent: MACs the conv2+fc layers would need if run frame-based for T steps."""
    c1, c2, fc = model.conv1, model.conv2, model.fc
    h1 = IMG_SIZE - c1.kernel_size[0] + 1
    enc = c1.out_channels * h1 * h1 * c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]
    h2 = h1 // 2 - c2.kernel_size[0] + 1
    conv2 = c2.out_channels * h2 * h2 * c2.in_channels * c2.kernel_size[0] * c2.kernel_size[1]
    return enc, model.num_steps * (conv2 + fc.in_features * fc.out_features)
