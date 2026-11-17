"""
cloud_api.py - cloud-tier "second opinion" backend.

    uvicorn cloud_api:app --host 0.0.0.0 --port 8000

POST /predict  accepts EITHER
    * multipart field `file`        : image file (png/jpg/...)
    * form field      `tensor_json` : JSON 2-D array [H][W] of raw 0-255 pixel values
Both are preprocessed for the bundled DenseNet121 ANN.
"""
import io
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import team_model as Model

STATE = {}
INFER_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE["model"] = Model.load_ann()
    print(f"[cloud] loaded DenseNet121 ANN from {Model.ANN_WEIGHTS_PATH.name}")
    yield
    STATE.clear()


app = FastAPI(title="Medical Imaging Kiosk - Cloud API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "DenseNet121-ANN"}


@app.post("/predict")
def predict(file: Optional[UploadFile] = File(None), tensor_json: Optional[str] = Form(None)):
    t0 = time.perf_counter()
    if file is None and tensor_json is None:
        raise HTTPException(status_code=400, detail="send `file` (image) or `tensor_json` (2-D array)")

    try:
        if file is not None:
            with Image.open(io.BytesIO(file.file.read())) as uploaded_image:
                img = uploaded_image.copy()
        else:
            arr = np.asarray(json.loads(tensor_json), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError("tensor_json must be a 2-D [H][W] array")
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    except Exception as exc:                               # bad upload / bad JSON
        raise HTTPException(status_code=422, detail=f"could not read image payload: {exc}")

    with INFER_LOCK:
        result = Model.predict_ann(STATE["model"], img)
    result["latency_ms"] = (time.perf_counter() - t0) * 1e3
    return {"status": "success", **result}
