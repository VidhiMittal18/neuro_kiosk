# Neuro Kiosk

Neuro Kiosk is a local research demonstration using a cloud DenseNet121 ANN prediction path with a local SNN fallback. It is not a medical device and must not be used to diagnose, screen, or guide care.

## Requirements

- Python 3.10–3.12 (3.11 recommended) and the dependencies in `requirements.txt`
- Node.js 18 or newer with npm for frontend development and production builds
- The bundled DenseNet121 ANN and SNN checkpoints and metadata under `models`, plus sample images

Recreate the Python environment (for example after clearing local generated files) from the project directory:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Restore the locked frontend dependencies:

```powershell
cd frontend
npm ci
```

## Development

Run the services in three terminals from the project root.

Terminal 1, cloud inference API:

```powershell
uvicorn cloud_api:app --host 127.0.0.1 --port 8000
```

Terminal 2, edge kiosk API:

```powershell
uvicorn edge_server:app --host 127.0.0.1 --port 8001
```

Terminal 3, frontend:

```powershell
cd frontend
npm run dev
```

Open the Vite URL printed in Terminal 3, normally `http://127.0.0.1:5173`. Vite proxies `/api/*` to the edge service and strips the `/api` prefix. The browser never calls the cloud API directly.

The frontend supports browser-camera capture and PNG/JPEG upload. Captures and diagnosis results are kept in the browser's IndexedDB review queue. Records remain pending until the edge API reports an online/cloud result; retries use Background Sync where supported and the browser's online event while the app is open otherwise.

## Production Build

Build the static frontend, then run only the edge service to serve both the UI and API on the same origin:

```powershell
cd frontend
npm run build
cd ..
uvicorn edge_server:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`. The edge service loads the bundled DenseNet121 SNN once at startup. This trusted checkpoint is a serialized SpikingJelly FX GraphModule; do not replace it with an untrusted `.pt` file. It checks the cloud DenseNet121 ANN API at `http://127.0.0.1:8000` by default; set `CLOUD_API_URL` to change that base URL. Start `cloud_api` separately when online inference is desired, and restart it after changing model code or weights because it loads the ANN once at startup. A valid cloud prediction is returned as-is, regardless of score; the edge SNN is used only when offline simulation is selected or the cloud/API request fails. The local CPU SNN runs for 16 timesteps using the package's validation-selected threshold.

## API

- `GET /status` reports the cloud model/reachability and whether the local SNN is loaded. The cloud API's `GET /health` identifies the loaded model as `DenseNet121-ANN`.
- `POST /diagnose` accepts multipart `file` and optional `simulate_offline` form fields.
- `GET /samples/sample_normal.png` and `GET /samples/sample_pneumonia.png` serve the bundled demonstration images.

Diagnosis uploads are limited to PNG/JPEG, 5 MiB, and 40 million decoded pixels. The endpoint returns consistent online/offline result data; local results include the SNN score, threshold, and timestep count. Keep the service bound to localhost or a trusted kiosk network; it has no authentication or user management.

## Checks

```powershell
pytest -q
cd frontend
npm test
npm run build
```

## Research and Data Limitations

The cloud path uses the package's DenseNet121 ANN and the local fallback uses its converted DenseNet121 SNN. Both use the same preprocessing: resize to 224 × 224, convert to three grayscale channels, and apply ImageNet normalization. The ANN uses its validation-selected threshold; the SNN uses its validation-selected threshold at 16 timesteps. These outputs are research scores, not clinically validated probabilities. The package reports 92.31% ANN and 88.14% SNN accuracy on the official test set, with the documented limitation that 64 images from that set had been examined during earlier experimentation. Results may not generalize beyond its pediatric AP chest X-ray scope.

Do not upload identifiable or otherwise sensitive medical images to this demonstration without appropriate authorization, security, privacy, and compliance review. The edge service is intended for a local trusted environment; do not expose it directly to the public internet.
