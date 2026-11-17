"""Local smoke test for explainability/xai_service.py.

Run from the Neuro Kiosk repository root:
    python explainability/test_xai_service.py --method grad2d

Other methods: grad3d, ig2d, ig3d, sam, inverted_sam, cam.
IG methods use 50 integration steps by default and can take longer on CPU.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image

from explainability.xai_service import DenseNetSNNXAI

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["grad2d", "grad3d", "ig2d", "ig3d", "sam", "inverted_sam", "cam"],
        default="grad2d",
    )
    parser.add_argument(
        "--sample",
        choices=["normal", "pneumonia"],
        default="pneumonia",
    )
    parser.add_argument("--timesteps", type=int, choices=[8, 16], default=16)
    parser.add_argument("--ig-steps", type=int, default=50)
    args = parser.parse_args()

    model_path = ROOT / "models" / "densenet121_pneumonia_snn.pt"
    metadata_path = ROOT / "models" / "metadata.json"
    sample_path = ROOT / "samples" / f"sample_{args.sample}.png"

    print("Project root:", ROOT)
    print("Model exists:", model_path.is_file())
    print("Metadata exists:", metadata_path.is_file())
    print("Sample exists:", sample_path.is_file())

    service = DenseNetSNNXAI(model_path, metadata_path, device="cpu")
    print("Service loaded:", service.health())

    with Image.open(sample_path) as source:
        image = source.copy()

    result = service.explain(
        image,
        method=args.method,
        timesteps=args.timesteps,
        ig_steps=args.ig_steps,
    )

    out_dir = ROOT / "explainability" / "results" / "service_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sample}_{args.method}_t{args.timesteps}"

    for key, suffix in [
        ("overlay_data_url", "overlay.png"),
        ("heatmap_data_url", "heatmap.png"),
        ("temporal_data_url", "temporal.png"),
    ]:
        data_url = result.get(key)
        if not data_url:
            continue
        encoded = data_url.split(",", 1)[1]
        destination = out_dir / f"{stem}_{suffix}"
        destination.write_bytes(base64.b64decode(encoded))
        print("Saved:", destination.relative_to(ROOT))

    print("Prediction:", result["prediction"])
    print("Pneumonia score:", result["pneumonia_score"])
    print("Threshold:", result["threshold"])
    print("Explanation target:", result["target_class"])
    print("Method:", result["method"])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
