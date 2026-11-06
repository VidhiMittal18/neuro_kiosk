#!/usr/bin/env python3
"""
train_and_benchmark.py - trains CNN baseline + SNN on PneumoniaMNIST, runs the 5-metric
edge benchmark, writes plots/JSON, exports weights and a few sample images for the kiosk.

    python train_and_benchmark.py                # defaults: 15 epochs each
    python train_and_benchmark.py --epochs 3     # quick run
"""
import argparse
import copy
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

import snntorch.functional as SF

import common as C

COL = {"CNN": "#1f77b4", "SNN": "#d62728"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_loaders(batch_size: int):
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tf = C.get_transform()                        # same transform object used at inference
    loaders = {}
    for split in ("train", "val", "test"):
        ds = PneumoniaMNIST(split=split, transform=tf, download=True, root=str(C.DATA_DIR))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
    return loaders


def export_samples():
    """Save one normal + one pneumonia test image as PNG for edge_app.py demos."""
    ds = PneumoniaMNIST(split="test", download=True, root=str(C.DATA_DIR))
    C.SAMPLES_DIR.mkdir(exist_ok=True)
    labels = ds.labels.reshape(-1)
    for cls, name in enumerate(C.CLASS_NAMES):
        idx = int(np.where(labels == cls)[0][0])
        Image.fromarray(ds.imgs[idx]).save(C.SAMPLES_DIR / f"sample_{name.lower()}.png")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(C.predict_probs(model, x.to(device)).argmax(1).cpu())
        ys.append(y.view(-1).long())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def fit(model, loaders, device, epochs, lr, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # SNN: rate-coded cross-entropy averaged over the T output spike frames.
    loss_fn = SF.ce_rate_loss() if model.is_spiking else nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_acc": []}
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, seen, t0 = 0.0, 0, time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.view(-1).long().to(device)
            opt.zero_grad(set_to_none=True)
            if model.is_spiking:
                spk_out, _ = model(x)             # BPTT through T steps with surrogate gradients
                loss = loss_fn(spk_out, y)
            else:
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        yt, yp = evaluate(model, loaders["val"], device)
        acc = accuracy_score(yt, yp)
        f1 = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2]
        hist["train_loss"].append(run_loss / seen)
        hist["val_acc"].append(100 * acc)
        if f1 > best_f1:                          # keep best checkpoint by validation macro-F1
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
        print(f"[{name}] epoch {ep:02d}/{epochs}  loss {run_loss / seen:.4f}  "
              f"val_acc {100 * acc:.2f}%  val_F1 {f1:.3f}  ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------- #
# Benchmark helpers
# --------------------------------------------------------------------------- #
def quality(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy_pct": 100 * accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f}


def measure_latency(model, sample, iters=100, warmup=10):
    """Mean/std ms per single image (batch=1) on CPU."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            C.predict_probs(model, sample)
        for _ in range(iters):
            t0 = time.perf_counter()
            C.predict_probs(model, sample)
            times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


def footprint(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_params, n_bytes / 1024 ** 2


def count_cnn_macs(model):
    """MACs of Conv2d + Linear layers for one 28x28 image (BN/ReLU/pool are negligible)."""
    total = [0]

    def conv_hook(m, _i, o):
        total[0] += (o.numel() // o.size(0)) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, _i, _o):
        total[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, C.IMG_SIZE, C.IMG_SIZE))
    for h in hooks:
        h.remove()
    return total[0]


@torch.no_grad()
def snn_activity_over_loader(model, loader):
    acc = C.new_activity_acc()
    for x, _ in loader:
        _, _, act = model.forward_record(x, record=True)
        C.accumulate_activity(model, act, acc)
    return C.summarize_activity(acc)


def run_benchmark(cnn, snn, loaders, iters):
    cnn_c, snn_c = copy.deepcopy(cnn).cpu().eval(), copy.deepcopy(snn).cpu().eval()
    test = loaders["test"]

    # (1) classification quality
    y_true, y_cnn = evaluate(cnn_c, test, "cpu")
    _, y_snn = evaluate(snn_c, test, "cpu")

    # (2) latency, batch size 1, CPU
    sample = next(iter(test))[0][:1]
    lat_cnn = measure_latency(cnn_c, sample, iters)
    lat_snn = measure_latency(snn_c, sample, iters)

    # (3) footprint
    p_cnn, m_cnn = footprint(cnn_c)
    p_snn, m_snn = footprint(snn_c)

    # (4) computational efficiency
    macs = count_cnn_macs(cnn_c)
    act = snn_activity_over_loader(snn_c, test)
    enc_macs, dense_ops = C.snn_static_ops(snn_c)

    # raster for the first test image (hidden layer 2)
    _, _, a1 = snn_c.forward_record(sample, record=True)
    raster = a1["lif2"][:, 0].reshape(snn_c.num_steps, -1).numpy()

    metrics = {
        "config": {"dataset": "PneumoniaMNIST", "num_steps": snn_c.num_steps, "beta": C.BETA,
                   "surrogate": f"fast_sigmoid(slope={C.SLOPE})", "test_samples": int(len(y_true)),
                   "latency_iterations": iters, "latency_device": "cpu"},
        "cnn": {
            "quality": quality(y_true, y_cnn),
            "latency_ms_per_image": {"mean": lat_cnn[0], "std": lat_cnn[1]},
            "footprint": {"trainable_params": p_cnn, "memory_mb": m_cnn},
            "compute": {"macs": macs, "flops": 2 * macs},
        },
        "snn": {
            "quality": quality(y_true, y_snn),
            "latency_ms_per_image": {"mean": lat_snn[0], "std": lat_snn[1]},
            "footprint": {"trainable_params": p_snn, "memory_mb": m_snn},
            "compute": {
                "sops_per_inference": act["sops_per_inference"],
                "sops_by_layer_per_inference": act["sops_by_layer_per_inference"],
                "encoding_macs_conv1": enc_macs,
                "dense_equivalent_synaptic_ops": dense_ops,
                "sops_to_dense_ratio": act["sops_per_inference"] / dense_ops,
                "sops_to_cnn_macs_ratio": act["sops_per_inference"] / macs,
            },
            "sparsity": {
                "mean_spike_rate_per_neuron_per_step": act["mean_spike_rate_per_neuron_per_step"],
                "per_layer_spike_rate": act["per_layer_spike_rate"],
                "spikes_per_inference": act["spikes_per_inference"],
            },
        },
    }
    extras = {"cm_cnn": confusion_matrix(y_true, y_cnn), "cm_snn": confusion_matrix(y_true, y_snn),
              "raster": raster}
    return metrics, extras


def summary_frame(m):
    c, s = m["cnn"], m["snn"]
    rows = {
        "Accuracy (%)": (f"{c['quality']['accuracy_pct']:.2f}", f"{s['quality']['accuracy_pct']:.2f}"),
        "Precision (macro)": (f"{c['quality']['precision_macro']:.3f}", f"{s['quality']['precision_macro']:.3f}"),
        "Recall (macro)": (f"{c['quality']['recall_macro']:.3f}", f"{s['quality']['recall_macro']:.3f}"),
        "F1 (macro)": (f"{c['quality']['f1_macro']:.3f}", f"{s['quality']['f1_macro']:.3f}"),
        "Latency (ms/img, CPU)": (f"{c['latency_ms_per_image']['mean']:.3f}", f"{s['latency_ms_per_image']['mean']:.3f}"),
        "Trainable params": (f"{c['footprint']['trainable_params']:,}", f"{s['footprint']['trainable_params']:,}"),
        "Memory (MB)": (f"{c['footprint']['memory_mb']:.4f}", f"{s['footprint']['memory_mb']:.4f}"),
        "MACs (CNN) / SOPs (SNN)": (f"{c['compute']['macs']:,.0f}", f"{s['compute']['sops_per_inference']:,.0f}"),
        "FLOPs (CNN, 2xMACs)": (f"{c['compute']['flops']:,.0f}", "-"),
        "Mean spike rate /neuron/step": ("-", f"{s['sparsity']['mean_spike_rate_per_neuron_per_step']:.4f}"),
    }
    return pd.DataFrame(rows, index=["CNN", "SNN"]).T


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(m, h_cnn, h_snn, ex, out_png):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.35)
    A = lambda r, c: fig.add_subplot(gs[r, c])
    c, s = m["cnn"], m["snn"]
    names = ["CNN", "SNN"]

    # training curves
    ax = A(0, 0)
    ax.plot(range(1, len(h_cnn["train_loss"]) + 1), h_cnn["train_loss"], color=COL["CNN"], label="CNN (cross-entropy)")
    ax.plot(range(1, len(h_snn["train_loss"]) + 1), h_snn["train_loss"], color=COL["SNN"], label="SNN (ce_rate_loss)")
    ax.set(title="Training loss", xlabel="epoch", ylabel="loss"); ax.legend()
    ax = A(0, 1)
    ax.plot(range(1, len(h_cnn["val_acc"]) + 1), h_cnn["val_acc"], color=COL["CNN"], label="CNN")
    ax.plot(range(1, len(h_snn["val_acc"]) + 1), h_snn["val_acc"], color=COL["SNN"], label="SNN")
    ax.set(title="Validation accuracy", xlabel="epoch", ylabel="%"); ax.legend()

    # raster
    ax = A(0, 2)
    t_idx, n_idx = np.nonzero(ex["raster"])
    ax.scatter(t_idx, n_idx, marker="|", s=18, color="black")
    ax.set(title="Spike raster - LIF2 (512 neurons), 1 test image", xlabel="time step", ylabel="neuron id",
           xlim=(-0.5, ex["raster"].shape[0] - 0.5), ylim=(-1, ex["raster"].shape[1]))

    # per-layer sparsity
    ax = A(0, 3)
    rates = [100 * s["sparsity"]["per_layer_spike_rate"][l] for l in C.LAYERS]
    ax.bar(["LIF1", "LIF2", "LIF3 (out)"], rates, color=COL["SNN"])
    ax.axhline(100 * s["sparsity"]["mean_spike_rate_per_neuron_per_step"], color="k", ls="--", label="overall mean")
    ax.set(title="Spike sparsity (test set)", ylabel="% neurons firing / step"); ax.legend()

    # quality bars
    ax = A(1, 0)
    df = pd.DataFrame([
        {"Metric": k, "Model": n, "Score": v}
        for n, q in (("CNN", c["quality"]), ("SNN", s["quality"]))
        for k, v in (("Accuracy", q["accuracy_pct"] / 100), ("Precision", q["precision_macro"]),
                     ("Recall", q["recall_macro"]), ("Macro F1", q["f1_macro"]))])
    sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette=COL, ax=ax)
    ax.set(title="Classification quality (test)", ylim=(0, 1.05))

    # latency / params / memory
    ax = A(1, 1)
    ax.bar(names, [c["latency_ms_per_image"]["mean"], s["latency_ms_per_image"]["mean"]],
           yerr=[c["latency_ms_per_image"]["std"], s["latency_ms_per_image"]["std"]],
           color=[COL[n] for n in names], capsize=6)
    ax.set(title="CPU latency / image (mean of 100)", ylabel="ms")
    ax = A(1, 2)
    ax.bar(names, [c["footprint"]["trainable_params"], s["footprint"]["trainable_params"]],
           color=[COL[n] for n in names])
    ax.set(title="Trainable parameters", ylabel="count")
    ax = A(1, 3)
    ax.bar(names, [c["footprint"]["memory_mb"], s["footprint"]["memory_mb"]], color=[COL[n] for n in names])
    ax.set(title="Model memory", ylabel="MB")

    # operations
    ax = A(2, 0)
    ops = {"CNN MACs": c["compute"]["macs"], "SNN SOPs\n(event-driven)": s["compute"]["sops_per_inference"],
           "SNN conv1\nencoding MACs": s["compute"]["encoding_macs_conv1"],
           "SNN dense-\nequivalent": s["compute"]["dense_equivalent_synaptic_ops"]}
    ax.bar(list(ops), list(ops.values()), color=[COL["CNN"], COL["SNN"], "#ff9896", "#aaaaaa"])
    ax.set(title="Ops per inference (log)", yscale="log", ylabel="operations")

    # confusion matrices
    for col, (key, nm) in zip((1, 2), (("cm_cnn", "CNN"), ("cm_snn", "SNN"))):
        ax = A(2, col)
        sns.heatmap(ex[key], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES)
        ax.set(title=f"{nm} confusion matrix", xlabel="predicted", ylabel="true")

    # text table
    ax = A(2, 3)
    ax.axis("off")
    ax.text(0, 1, summary_frame(m).to_string(), family="monospace", fontsize=8.5, va="top")
    ax.set_title("Summary")

    fig.suptitle("PneumoniaMNIST - CNN baseline vs Spiking CNN (edge benchmark)", fontsize=18)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-cnn", type=float, default=1e-3)
    ap.add_argument("--lr-snn", type=float, default=1e-3)
    ap.add_argument("--latency-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    loaders = get_loaders(args.batch_size)
    cnn, snn = C.CNNBaseline(), C.SpikingCNN()
    h_cnn = fit(cnn, loaders, device, args.epochs, args.lr_cnn, "CNN")
    h_snn = fit(snn, loaders, device, args.epochs, args.lr_snn, "SNN")

    torch.save(cnn.cpu().state_dict(), C.CNN_PATH)
    torch.save(snn.cpu().state_dict(), C.SNN_PATH)
    export_samples()

    metrics, extras = run_benchmark(cnn, snn, loaders, args.latency_iters)
    with open(C.ROOT / "benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    make_plots(metrics, h_cnn, h_snn, extras, C.ROOT / "benchmark_results.png")

    print("\n" + summary_frame(metrics).to_string())
    print(f"\nSaved: {C.CNN_PATH.name}, {C.SNN_PATH.name}, benchmark_metrics.json, "
          f"benchmark_results.png, samples/")


if __name__ == "__main__":
    main()
