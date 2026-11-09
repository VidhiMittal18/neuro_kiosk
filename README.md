# Neuro Kiosk

Neuro Kiosk is a local research demonstration for comparing a cloud CNN prediction path with an edge SNN fallback. It is not a medical device and must not be used to diagnose, screen, or guide care.

## Requirements

- Python 3.10 or newer and the dependencies in `requirements.txt`
- Node.js 18 or newer with npm for frontend development and production builds
- The checked-in `trained_cnn.pt`, `trained_snn.pt`, benchmark assets, and sample images

Install the Python dependencies from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install the frontend dependencies once:

```powershell
cd frontend
npm install
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

## Production Build

Build the static frontend, then run only the edge service to serve both the UI and API on the same origin:

```powershell
cd frontend
npm run build
cd ..
uvicorn edge_server:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`. The edge service loads the SNN once at startup. It checks the cloud API at `http://127.0.0.1:8000` by default; set `CLOUD_API_URL` to change that base URL. Start `cloud_api` separately when online inference is desired. If the cloud is unavailable or offline simulation is enabled, the edge service runs the local CPU SNN.

## API

- `GET /status` reports cloud reachability and whether the local SNN is loaded.
- `POST /diagnose` accepts multipart `file` and optional `simulate_offline` form fields.
- `GET /benchmarks` returns `benchmark_metrics.json`.
- `GET /benchmark-image` serves `benchmark_results.png`.
- `GET /samples/sample_normal.png` and `GET /samples/sample_pneumonia.png` serve the bundled demonstration images.

Diagnosis uploads are limited to PNG/JPEG, 5 MiB, and 40 million decoded pixels. The endpoint returns consistent online/offline result data and SNN telemetry for local inference. Keep the service bound to localhost or a trusted kiosk network; it has no authentication or user management.

## Checks

```powershell
pytest -q
cd frontend
npm test
npm run build
```

## Research and Data Limitations

The checked-in models were trained on PneumoniaMNIST and classify only `Normal` or `Pneumonia`. Input images are converted to grayscale and resized to 28 × 28 pixels. This small research dataset and preprocessing are not validated for real clinical images or clinical use. The displayed confidence is a model score, not a probability that should guide care.

Benchmark operation counts are estimates: CNN MACs and SNN SOPs are different operation types and are not direct energy measurements. The recorded CPU benchmark shows higher SNN inference latency; this kiosk does not claim that SNN CPU latency is an advantage.

Do not upload identifiable or otherwise sensitive medical images to this demonstration without appropriate authorization, security, privacy, and compliance review. The edge service is intended for a local trusted environment; do not expose it directly to the public internet.