#!/usr/bin/env python3
"""
edge_app.py - local diagnostic kiosk client.

    python edge_app.py samples/sample_pneumonia.png
    python edge_app.py my_xray.png --url http://localhost:8000/predict --timeout 2

Routing logic
  1. ONLINE : POST the image to the cloud API (2 s timeout) and print its "second opinion".
  2. OFFLINE: any requests.exceptions.RequestException (timeout, connection refused, DNS, HTTP
              error, ...) is caught -> alert -> load trained_snn.pt -> on-device CPU inference
              with the low-power SNN -> print local diagnostic metrics. The kiosk never crashes
              because of the network.
"""
import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image

import common as C

CLOUD_URL = "http://localhost:8000/predict"
TIMEOUT_S = 2


def query_cloud(image_bytes: bytes, filename: str, url: str, timeout: float) -> dict:
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    resp = requests.post(url, files={"file": (filename, image_bytes, mime)}, timeout=timeout)
    resp.raise_for_status()                       # 4xx/5xx -> HTTPError (a RequestException)
    return resp.json()


def local_snn_diagnosis(image_path: Path) -> dict:
    """On-device CPU inference with the SNN (same preprocessing + decision rule as everywhere)."""
    t_load = time.perf_counter()
    model = C.load_trained("snn", C.SNN_PATH, device="cpu")
    load_ms = (time.perf_counter() - t_load) * 1e3

    t0 = time.perf_counter()
    x = C.preprocess_pil(Image.open(image_path))
    with torch.no_grad():
        spk_out, mem_out, act = model.forward_record(x, record=True)
        probs = torch.softmax(C.snn_scores(spk_out, mem_out), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1e3

    idx = int(probs.argmax())
    acc = C.new_activity_acc()
    C.accumulate_activity(model, act, acc)
    stats = C.summarize_activity(acc)
    _, dense_ops = C.snn_static_ops(model)
    return {
        "status": "success",
        "source": "Edge-SNN (local CPU)",
        "prediction": C.CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "latency_ms": latency_ms,
        "model_load_ms": load_ms,
        "time_steps": model.num_steps,
        "output_spike_counts": {C.CLASS_NAMES[i]: int(spk_out[:, 0, i].sum()) for i in range(C.NUM_CLASSES)},
        "spikes_this_inference": stats["spikes_per_inference"],
        "mean_spike_rate_per_neuron_per_step": stats["mean_spike_rate_per_neuron_per_step"],
        "per_layer_spike_rate": stats["per_layer_spike_rate"],
        "sops_this_inference": stats["sops_per_inference"],
        "dense_equivalent_ops": dense_ops,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge diagnostic kiosk")
    ap.add_argument("image", type=Path, help="path to the X-ray image")
    ap.add_argument("--url", default=CLOUD_URL)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args()

    if not args.image.is_file():
        print(f"[kiosk] image not found: {args.image}")
        return 2
    image_bytes = args.image.read_bytes()

    print(f"[kiosk] analysing {args.image.name} ...")
    try:
        # ---- ONLINE PATH ---------------------------------------------------------------
        result = query_cloud(image_bytes, args.image.name, args.url, args.timeout)
        print("[kiosk] ONLINE - cloud second opinion received:")
        print(json.dumps(result, indent=2))
        return 0
    except (requests.exceptions.RequestException, ValueError) as exc:
        # ---- FALLBACK PATH -------------------------------------------------------------
        # RequestException covers Timeout, ConnectionError (refused/DNS), HTTPError, etc.
        # ValueError covers a malformed (non-JSON) response on older `requests` versions.
        print("\n" + "!" * 64)
        print("  OFFLINE ALERT: cloud unreachable -> switching to on-device SNN")
        print(f"  reason: {type(exc).__name__}: {exc}")
        print("!" * 64 + "\n")

    try:
        local = local_snn_diagnosis(args.image)
    except FileNotFoundError as exc:
        print(f"[kiosk] cannot run offline fallback: {exc}")
        return 3
    except Exception as exc:                      # unreadable image, etc.
        print(f"[kiosk] local inference failed: {exc}")
        return 4

    print("[kiosk] OFFLINE - local diagnostic result:")
    print(json.dumps(local, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
