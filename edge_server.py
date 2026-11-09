"""Edge kiosk API: cloud inference with a local SNN fallback."""
from __future__ import annotations

import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import common as C


ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 40_000_000
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
CLOUD_TIMEOUT_S = 2.0
BENCHMARK_PATH = ROOT / "benchmark_metrics.json"
BENCHMARK_IMAGE_PATH = ROOT / "benchmark_results.png"
SAMPLE_PATHS = {
    "sample_normal.png": ROOT / "samples" / "sample_normal.png",
    "sample_pneumonia.png": ROOT / "samples" / "sample_pneumonia.png",
}
FRONTEND_DIST = ROOT / "frontend" / "dist"
INFER_LOCK = threading.Lock()


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


class DiagnoseBodyLimitMiddleware:
    """Keep oversized multipart requests out of Starlette's form parser."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/diagnose":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    response = _json_error(422, "Image must be 5 MB or smaller.")
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = _json_error(400, "Invalid Content-Length header.")
                await response(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BYTES:
                response = _json_error(422, "Image must be 5 MB or smaller.")
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        sent = False

        async def replay_body() -> dict:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)


def _load_image(image_bytes: bytes, content_type: str) -> Image.Image:
    expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}.get(content_type.lower())
    if expected_format is None:
        raise HTTPException(status_code=422, detail="Upload a PNG or JPEG image.")
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != expected_format:
                raise HTTPException(status_code=422, detail="Image content does not match its file type.")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=422, detail="Image dimensions are too large.")
            image.verify()
        return Image.open(io.BytesIO(image_bytes)).convert("L")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from None


async def _read_upload(file: UploadFile) -> tuple[bytes, Image.Image]:
    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail="Image must be 5 MB or smaller.")
    image = _load_image(image_bytes, file.content_type or "")
    return image_bytes, image


def _local_snn_result(image: Image.Image, model: C.SpikingCNN) -> dict:
    started = time.perf_counter()
    tensor = C.preprocess_pil(image)
    with INFER_LOCK, torch.no_grad():
        spike_output, membrane_output, activity = model.forward_record(tensor, record=True)
        probabilities = torch.softmax(C.snn_scores(spike_output, membrane_output), dim=1)[0]
    latency_ms = (time.perf_counter() - started) * 1000
    prediction_index = int(probabilities.argmax())
    accumulator = C.new_activity_acc()
    C.accumulate_activity(model, activity, accumulator)
    stats = C.summarize_activity(accumulator)
    _, dense_ops = C.snn_static_ops(model)
    return {
        "prediction": C.CLASS_NAMES[prediction_index],
        "confidence": float(probabilities[prediction_index]),
        "latency_ms": latency_ms,
        "source": "Edge-SNN (local CPU)",
        "snn": {
            "time_steps": model.num_steps,
            "output_spike_counts": {
                C.CLASS_NAMES[index]: int(spike_output[:, 0, index].sum())
                for index in range(C.NUM_CLASSES)
            },
            "spikes_this_inference": stats["spikes_per_inference"],
            "mean_spike_rate_per_neuron_per_step": stats["mean_spike_rate_per_neuron_per_step"],
            "per_layer_spike_rate": stats["per_layer_spike_rate"],
            "sops_this_inference": stats["sops_per_inference"],
            "dense_equivalent_ops": dense_ops,
        },
    }


def _cloud_result(image_bytes: bytes, filename: str) -> dict:
    response = requests.post(
        f"{CLOUD_API_URL}/predict",
        files={"file": (filename, image_bytes)},
        timeout=CLOUD_TIMEOUT_S,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("prediction") not in C.CLASS_NAMES:
        raise ValueError("Cloud API returned an invalid prediction.")
    return {
        "prediction": result["prediction"],
        "confidence": float(result["confidence"]),
        "latency_ms": float(result["latency_ms"]),
        "source": result.get("source", "Cloud-API"),
    }


def _offline_reason(error: Exception) -> str:
    if isinstance(error, requests.exceptions.Timeout):
        return f"{type(error).__name__}: cloud API timed out after {CLOUD_TIMEOUT_S:g} seconds."
    if isinstance(error, requests.exceptions.HTTPError):
        status = error.response.status_code if error.response is not None else "unknown"
        return f"HTTPError: cloud API returned HTTP {status}."
    if isinstance(error, requests.exceptions.RequestException):
        return f"{type(error).__name__}: cloud API could not be reached."
    return "InvalidResponse: cloud API returned an invalid prediction response."


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.snn_model = C.load_trained("snn", C.SNN_PATH, device="cpu")
    yield
    application.state.snn_model = None


app = FastAPI(title="Neuro Kiosk Edge API", lifespan=lifespan)
app.state.snn_model = None
app.add_middleware(DiagnoseBodyLimitMiddleware)


@app.get("/status")
def status() -> dict:
    try:
        response = requests.get(f"{CLOUD_API_URL}/health", timeout=1.0)
        cloud_reachable = response.ok
    except requests.exceptions.RequestException:
        cloud_reachable = False
    return {"cloud_reachable": cloud_reachable, "snn_loaded": app.state.snn_model is not None}


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), simulate_offline: bool = Form(False)) -> dict:
    image_bytes, image = await _read_upload(file)
    if simulate_offline:
        result = _local_snn_result(image, app.state.snn_model)
        return {"mode": "offline", "offline_reason": "Simulated offline", "result": result}

    try:
        result = _cloud_result(image_bytes, file.filename or "upload")
        return {"mode": "online", "offline_reason": None, "result": result}
    except (requests.exceptions.RequestException, ValueError, TypeError) as error:
        result = _local_snn_result(image, app.state.snn_model)
        return {"mode": "offline", "offline_reason": _offline_reason(error), "result": result}


@app.get("/benchmarks")
def benchmarks() -> dict:
    try:
        return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark metrics are unavailable.") from None
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Benchmark metrics could not be read.") from None


@app.get("/benchmark-image")
def benchmark_image() -> FileResponse:
    if not BENCHMARK_IMAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Benchmark image is unavailable.")
    return FileResponse(BENCHMARK_IMAGE_PATH, media_type="image/png", filename=BENCHMARK_IMAGE_PATH.name)


@app.get("/samples/{sample_name}")
def sample_image(sample_name: str) -> FileResponse:
    sample_path = SAMPLE_PATHS.get(sample_name)
    if sample_path is None or not sample_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(sample_path, media_type="image/png", filename=sample_path.name)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")