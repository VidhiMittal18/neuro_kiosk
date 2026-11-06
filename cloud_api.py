"""
cloud_api.py - cloud-tier "second opinion" backend.

    uvicorn cloud_api:app --host 0.0.0.0 --port 8000
    CLOUD_MODEL=snn uvicorn cloud_api:app      # serve the SNN instead (default: cnn)

POST /predict  accepts EITHER
    * multipart field `file`        : image file (png/jpg/...)
    * form field      `tensor_json` : JSON 2-D array [H][W] of raw 0-255 pixel values
Both are converted to a PIL image and pushed through common.preprocess_pil(), i.e. the exact
pipeline used in training.
"""
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import common as C

MODEL_KIND = os.getenv("CLOUD_MODEL", "cnn").lower()
STATE = {}
# snnTorch Leaky layers keep their membrane in module attributes, so concurrent requests
# (FastAPI runs sync endpoints in a thread pool) must not interleave on one model instance.
INFER_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    path = C.CNN_PATH if MODEL_KIND == "cnn" else C.SNN_PATH
    STATE["model"] = C.load_trained(MODEL_KIND, path, "cpu")
    print(f"[cloud] loaded {MODEL_KIND.upper()} from {path.name}")
    yield
    STATE.clear()


app = FastAPI(title="Medical Imaging Kiosk - Cloud API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND}


@app.post("/predict")
def predict(file: Optional[UploadFile] = File(None), tensor_json: Optional[str] = Form(None)):
    t0 = time.perf_counter()
    if file is None and tensor_json is None:
        raise HTTPException(status_code=400, detail="send `file` (image) or `tensor_json` (2-D array)")

    try:
        if file is not None:
            img = Image.open(io.BytesIO(file.file.read()))
        else:
            arr = np.asarray(json.loads(tensor_json), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError("tensor_json must be a 2-D [H][W] array")
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        x = C.preprocess_pil(img)                          # identical to training pipeline
    except Exception as exc:                               # bad upload / bad JSON
        raise HTTPException(status_code=422, detail=f"could not read image payload: {exc}")

    with INFER_LOCK, torch.no_grad():
        probs = C.predict_probs(STATE["model"], x)[0]
    idx = int(probs.argmax())
    return {
        "status": "success",
        "source": "Cloud-API",
        "prediction": C.CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "latency_ms": (time.perf_counter() - t0) * 1e3,
    }
