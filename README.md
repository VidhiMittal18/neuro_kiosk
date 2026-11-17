# Neuro Kiosk

**Edge-deployable pneumonia screening demo using a spiking neural network (SNN) fallback.**

Neuro Kiosk is a local research demonstration. A kiosk captures or uploads a chest X-ray, sends it to a cloud DenseNet121 ANN for prediction, and, if the cloud is unreachable, falls back to a DenseNet121 SNN running locally on CPU. A built-in explainability service can show which image regions drove the SNN's decision.

> **Not a medical device.** This project must not be used to diagnose, screen, or guide care. Outputs are research scores, not clinically validated probabilities.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Models and Results](#models-and-results)
- [Model Input](#model-input)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Limitations](#limitations)
- [Security and Privacy](#security-and-privacy)
- [License](#license)

---

## How It Works

```
 ┌──────────────┐   /api/*    ┌─────────────────────┐   POST /predict   ┌──────────────────────┐
 │  React/Vite  │ ──────────► │   Edge API (:8001)  │ ────────────────► │  Cloud API (:8000)   │
 │  frontend    │ ◄────────── │   edge_server.py    │ ◄──────────────── │  DenseNet121 ANN     │
 └──────────────┘             │                     │   (2 s timeout)   └──────────────────────┘
                              │   cloud fails or    │
                              │   offline selected  │
                              │          │          │
                              │          ▼          │
                              │  Local SNN fallback │
                              │  DenseNet121 SNN    │
                              │  16 timesteps, CPU  │
                              │                     │
                              │  /explain → XAI     │
                              └─────────────────────┘
```

1. The browser captures an image (camera) or uploads a PNG/JPEG. The browser never calls the cloud API directly; Vite proxies `/api/*` to the edge service.
2. The edge API validates the image and forwards it to the cloud ANN.
3. A valid cloud prediction is returned as-is, regardless of score.
4. If the cloud request times out (2 seconds), fails, or returns an invalid response, or if offline simulation is selected, the local SNN produces the prediction instead.
5. Results are stored in the browser's IndexedDB review queue. Records stay pending until the edge API reports an online/cloud result. Retries use Background Sync where supported, and the browser's `online` event otherwise.

## Models and Results

| Path | Model | Role |
|------|-------|------|
| Cloud | DenseNet121 ANN | Primary prediction when online |
| Edge | DenseNet121 SNN (converted from the ANN, SpikingJelly, 16 timesteps) | Offline fallback and explainability |

Accuracy on the official test set, as reported by the bundled package:

| Model | Accuracy |
|-------|----------|
| DenseNet121 ANN | **92.31%** |
| DenseNet121 SNN (16 timesteps) | **88.14%** |

Both models use a decision threshold selected on the validation set (stored in `models/metadata.json`). The SNN produces its score by summing logits over all timesteps, averaging them, and applying a sigmoid.

**Caveat:** 64 images from the test set were examined during earlier experimentation, so these figures may be slightly optimistic.

## Model Input

Both models share the same preprocessing:

| Step | Detail |
|------|--------|
| Resize | 224 × 224 (aspect ratio not preserved) |
| Channels | Grayscale, replicated to 3 channels |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Tensor shape | `(1, 3, 224, 224)` (batch, channels, height, width) |

The SNN receives the same tensor at each of its 16 timesteps. Class mapping is `NORMAL = 0`, `PNEUMONIA = 1`; the output is a single logit passed through a sigmoid.

## Project Structure

```
neuro_kiosk/
├── cloud_api.py         # Cloud inference API (DenseNet121 ANN), port 8000
├── edge_server.py       # Edge kiosk API: cloud-first, SNN fallback, XAI; serves built frontend
├── local_snn.py         # Local SNN loading and inference (SpikingJelly)
├── team_model.py        # Shared preprocessing, metadata loading, ANN loading and inference
├── explainability/      # XAI service for the DenseNet121 SNN
├── frontend/            # React + Vite kiosk UI (camera capture, upload, review queue)
├── models/              # ANN weights, SNN checkpoint, metadata.json
├── samples/             # sample_normal.png, sample_pneumonia.png
├── tests/               # Python tests (pytest)
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

## Requirements

- Python 3.10–3.12 (3.11 recommended)
- Node.js 18 or newer with npm
- The bundled DenseNet121 ANN and SNN checkpoints and metadata under `models/`, plus the sample images

## Setup

Create the Python environment from the project directory.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

**Linux / macOS:**

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install the locked frontend dependencies:

```bash
cd frontend
npm ci
```

## Running the Project

### Development (three terminals, from the project root)

**Terminal 1: cloud inference API**

```bash
uvicorn cloud_api:app --host 127.0.0.1 --port 8000
```

**Terminal 2: edge kiosk API**

```bash
uvicorn edge_server:app --host 127.0.0.1 --port 8001
```

**Terminal 3: frontend**

```bash
cd frontend
npm run dev
```

Open the Vite URL printed in Terminal 3 (normally `http://127.0.0.1:5173`).

### Production build

Build the static frontend, then run only the edge service. It serves both the UI and the API on one origin.

```bash
cd frontend
npm run build
cd ..
uvicorn edge_server:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`.

- The edge service loads the SNN once at startup.
- It checks the cloud API at `http://127.0.0.1:8000` by default. Set the `CLOUD_API_URL` environment variable to change this.
- Start `cloud_api` separately when online inference is wanted, and restart it after changing model code or weights (it loads the ANN once at startup).

### Trying the offline fallback

Select offline simulation in the UI (or send `simulate_offline=true` to `/diagnose`) to force the local SNN, or simply stop the cloud API.

## API Reference

### Edge API (port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Cloud reachability, cloud model name, and whether the local SNN is loaded |
| `POST` | `/diagnose` | Multipart `file` (required) and `simulate_offline` (optional, default `false`). Returns the prediction and the mode (`online` or `offline`) |
| `POST` | `/explain` | Generates an explainability visualization from the DenseNet121 SNN (see below) |
| `GET` | `/xai/health` | Reports whether the explainability service is loaded |
| `GET` | `/samples/sample_normal.png` | Bundled normal sample image |
| `GET` | `/samples/sample_pneumonia.png` | Bundled pneumonia sample image |

**`POST /diagnose` response**

```json
{
  "mode": "online | offline",
  "offline_reason": null,
  "result": {
    "prediction": "Normal | Pneumonia",
    "confidence": 0.0,
    "pneumonia_score": 0.0,
    "threshold": 0.0,
    "latency_ms": 0.0,
    "source": "Cloud-DenseNet121-ANN | Edge-SNN (local CPU)"
  }
}
```

Local (SNN) results also include `time_steps`.

**`POST /explain` form fields**

| Field | Default | Description |
|-------|---------|-------------|
| `file` | required | PNG or JPEG image |
| `method` | `grad2d` | Explanation method (see `explainability/`) |
| `timesteps` | `16` | Number of SNN timesteps |
| `target_class` | `predicted` | `predicted`, `NORMAL`, or `PNEUMONIA` |
| `ig_steps` | `50` | Steps for integration-based methods |

### Cloud API (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{"status": "ok", "model": "DenseNet121-ANN"}` |
| `POST` | `/predict` | Accepts either multipart `file` (an image) or form field `tensor_json` (a 2-D `[H][W]` array of raw 0–255 pixel values) |

### Upload limits

- PNG or JPEG only, and the file content must match its declared type
- Maximum 5 MiB per image
- Maximum 40 million decoded pixels
- Oversized or invalid uploads return HTTP 422

## Testing

```bash
pytest -q
cd frontend
npm test
npm run build
```

## Limitations

- **Research demo only.** Not clinically validated and not for diagnosis, screening, or care decisions.
- **Narrow scope.** Results may not generalize beyond the pediatric AP chest X-ray scope of the training data.
- **Converted SNN.** The SNN is converted from a trained ANN rather than trained directly as a spiking network. It is about 4 percentage points less accurate than the ANN (88.14% vs 92.31%).
- **Test-set exposure.** 64 test images were examined during earlier experimentation.
- **Edge cost.** A DenseNet121 SNN run for 16 timesteps on CPU is heavier than a typical lightweight edge model.
- **Scores are not probabilities.** Outputs are research scores with validation-selected thresholds.

## Security and Privacy

- Keep the services bound to localhost or a trusted kiosk network. The API has **no authentication or user management**, and the edge service must not be exposed directly to the public internet.
- Do not upload identifiable or otherwise sensitive medical images without appropriate authorization, security, privacy, and compliance review.
- The SNN checkpoint is a serialized SpikingJelly FX `GraphModule` loaded with `torch.load(..., weights_only=False)`. **Never replace it with an untrusted `.pt` file.**
- Error responses from the explainability endpoint omit internal details; they are logged server-side.

## License

Released under the MIT License. See [LICENSE](LICENSE).
