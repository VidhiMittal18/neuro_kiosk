# NeuroKiosk — Edge-Deployable Pneumonia Screening with a Spiking Neural Network

A prototype diagnostic kiosk that classifies chest X-rays as **Normal** or **Pneumonia**.
It asks a cloud model for a "second opinion" when the network is available, and **automatically falls back to a
lightweight Spiking Neural Network (SNN) running on the device** when it is not. The repo also benchmarks the SNN
against an equivalent conventional CNN to study the accuracy / latency / compute trade-offs of spiking models for
low-power edge deployment.

> **Research prototype only. Not a medical device and not for clinical use.**

---

## 1. Motivation

Clinics in remote or low-connectivity areas cannot rely on a cloud service being reachable. A diagnostic kiosk
therefore needs to:

1. use a strong cloud model when the network is up, and
2. keep working, on small and low-power hardware, when it is not.

Spiking Neural Networks are a candidate for the second case: they communicate with sparse binary events (spikes)
instead of dense floating-point activations, which maps well onto event-driven, low-power neuromorphic hardware.
This project builds the full pipeline (training → benchmark → cloud API → offline-capable edge client) and measures
honestly how a small SNN compares with a small CNN on the same task.

---

## 2. System overview

```
                        ┌────────────────────────────────┐
                        │        Cloud API (FastAPI)     │
                        │   CNN by default (SNN optional)│
                        └──────────────▲─────────────────┘
                                       │  POST /predict  (2 s timeout)
   X-ray image                         │
 ┌───────────────┐   ┌─────────────────┴──────────┐
 │  edge_app.py  │──▶│   Network reachable?        │
 │  (the kiosk)  │   └───────┬─────────────┬───────┘
 └───────────────┘        yes│             │no / timeout / error
                              ▼             ▼
                    print cloud        OFFLINE ALERT
                    "second opinion"   load trained_snn.pt
                                       run SNN on local CPU
                                       print prediction + spike statistics
```

Training, the cloud API and the edge app all import **`common.py`**, so preprocessing, model definitions and the
decision rule are identical in every runtime.

---

## 3. Repository layout

| File | Purpose |
|---|---|
| `common.py` | Single source of truth: preprocessing, `CNNBaseline`, `SpikingCNN`, scoring rule, spike/SOP accounting |
| `train_and_benchmark.py` | Trains both models on PneumoniaMNIST, runs the benchmark, writes metrics + plots + weights + sample images |
| `cloud_api.py` | FastAPI service exposing `/health` and `/predict` |
| `edge_app.py` | Kiosk client: tries the cloud, falls back to the on-device SNN |
| `trained_cnn.pt`, `trained_snn.pt` | Pre-trained weights (no training needed to run the demo) |
| `benchmark_metrics.json`, `benchmark_results.png` | Saved benchmark output |
| `samples/` | One normal and one pneumonia test image for the demo |
| `requirements.txt` | Python dependencies |

---

## 4. Dataset

**PneumoniaMNIST** (from [MedMNIST](https://medmnist.com/)): chest X-rays downsampled to **28 × 28 grayscale**,
binary label (0 = Normal, 1 = Pneumonia). The pre-defined train / validation / test splits are used; the test split
has 624 images (234 Normal, 390 Pneumonia).

Preprocessing (`common.get_transform`): convert to 1-channel, resize to 28 × 28 if needed, scale to [0, 1], normalise
to [-1, 1]. The same transform is used in training, in the API and on the kiosk, so arbitrary uploaded images are
handled consistently.

---

## 5. Models

### 5.1 CNN baseline (`CNNBaseline`)
`Conv(1→16) → BN → ReLU → MaxPool → Conv(16→32) → BN → ReLU → MaxPool → Linear(1568→2)` — 8,034 parameters.
Trained with cross-entropy.

### 5.2 Spiking CNN (`SpikingCNN`)
```
image ─ conv1(1→12, 5×5) ─ pool ─▶ [LIF1] ─ conv2(12→32, 5×5) ─ pool ─▶ [LIF2] ─ fc(512→2) ─▶ [LIF3] ─▶ output spikes
```
10,970 parameters. Built with [snnTorch](https://snntorch.readthedocs.io/).

* **Neuron model — Leaky Integrate-and-Fire (LIF).** Each neuron keeps a membrane potential `U`. At every time step
  it leaks (`× β`), adds the incoming current, and emits a binary spike when `U` crosses the threshold, after which
  the threshold is subtracted (soft reset):
  `U[t] = β·U[t-1] + I[t] − S[t-1]·threshold`, `S[t] = 1 if U[t] > threshold else 0`.
  Here `β = 0.9`, threshold = 1.0 (snnTorch default).
* **Time.** The network is simulated for `T = 15` time steps per image.
* **Input encoding — direct encoding.** The analog image is fed as a constant current, so `conv1` is computed once
  and reused every step. Everything after `conv1` receives binary spikes.
* **Output decoding — rate coding.** The predicted class is the output neuron that fires the most spikes over the
  15 steps (a tiny membrane-potential term breaks ties). Softmax over the spike counts gives the reported confidence.
* **Training — surrogate gradients + BPTT.** A spike is a step function, whose derivative is zero almost everywhere,
  so ordinary back-propagation cannot train it. In the forward pass the exact step function is kept; in the backward
  pass a smooth `fast_sigmoid(slope=25)` derivative is used instead. Gradients are propagated through all 15 time
  steps (back-propagation through time). Loss: `ce_rate_loss` (cross-entropy on output spike counts). Optimiser: Adam,
  lr 1e-3. The best epoch is chosen by validation macro-F1.

What is learned: the convolution and linear **weights and biases**. `β` and the threshold are fixed hyper-parameters
in this version.

---

## 6. Benchmark results

Test set, 624 images. CPU latency is the mean of 100 single-image runs (batch size 1).

| Metric | CNN | SNN |
|---|---|---|
| Accuracy | 86.22 % | 86.06 % |
| Precision (macro) | 0.904 | 0.890 |
| Recall (macro) | 0.818 | 0.821 |
| F1 (macro) | 0.839 | 0.839 |
| Pneumonia caught (true positives) | 388 / 390 | 382 / 390 |
| Normal correctly cleared (true negatives) | 150 / 234 | 155 / 234 |
| Trainable parameters | 8,034 | 10,970 |
| Model memory | 0.031 MB | 0.050 MB |
| CPU latency / image | 0.41 ms | 24.0 ms |
| Compute per inference | 1,019,200 MACs | 1,703,640 SOPs (+ 172,800 MACs for conv1 encoding) |

SNN activity (test set): on average **23 %** of neurons fire per time step (LIF1 18.5 %, LIF2 38.3 %, output 48.4 %),
about **7,760 spikes per inference**. Synaptic operations are only **18 %** of what the same layers would cost if run
densely for 15 steps (1.70 M vs 9.23 M). Full plots: `benchmark_results.png`.

### How to read these results honestly

* **Accuracy parity.** The SNN matches the CNN on every quality metric (within ~0.2 pp accuracy).
* **No speed or op-count win on a normal CPU.** Simulating 15 time steps on a CPU is ~59× slower than a single CNN
  forward pass, and the SNN needs more parameters and more operations than the CNN. The expected advantage of an SNN
  comes from **event-driven neuromorphic hardware**, where a spike-triggered accumulate is cheaper than a
  multiply-accumulate and silent neurons cost nothing. The sparsity numbers above are the evidence that such hardware
  could exploit; they are not a measured energy saving.
* **Operations are not the same as energy.** SOPs and MACs are different kinds of operation; the comparison here is
  a proxy, not a measurement on real hardware.
* **Class imbalance in errors.** Both models are very sensitive to pneumonia (almost no missed cases) but clear only
  ~64–66 % of normal scans, i.e. they raise many false alarms. Validation accuracy during training is ~95–98 %,
  noticeably higher than test accuracy, which reflects a known distribution gap between the PneumoniaMNIST splits.
* **28 × 28 images** are far below clinical resolution; results are a proof of concept for the pipeline, not
  evidence of diagnostic performance.

---

## 7. Getting started

```bash
git clone https://github.com/VidhiMittal18/neuro_kiosk
cd neuro_kiosk
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the demo (pre-trained weights are included)

```bash
# Terminal 1 — cloud backend
uvicorn cloud_api:app --host 0.0.0.0 --port 8000

# Terminal 2 — kiosk, ONLINE path (cloud answers)
python edge_app.py samples/sample_pneumonia.png

# Stop the server in Terminal 1 (Ctrl+C), then run again — OFFLINE path
python edge_app.py samples/sample_pneumonia.png
```

Offline output shows an alert, then the local SNN result: prediction, confidence, latency, output spike counts per
class, spikes in this inference, per-layer spike rate, and SOPs vs the dense-equivalent cost.

Options: `--url http://<host>:8000/predict`, `--timeout 2` (seconds before falling back).
To serve the SNN from the cloud instead of the CNN: `CLOUD_MODEL=snn uvicorn cloud_api:app`.

### Re-train and re-benchmark

```bash
python train_and_benchmark.py              # 15 epochs each (downloads PneumoniaMNIST to ./data)
python train_and_benchmark.py --epochs 3   # quick run
```
This regenerates the weights, `benchmark_metrics.json`, `benchmark_results.png` and `samples/`.

### API

`GET /health` → `{"status": "ok", "model": "cnn"}`

`POST /predict` — send **either** a multipart `file` (image) **or** a form field `tensor_json` (2-D array of 0–255
pixel values).
```json
{"status": "success", "source": "Cloud-API", "prediction": "Pneumonia", "confidence": 0.99, "latency_ms": 3.2}
```

---

## 8. Design notes

* **Fail-safe routing.** Any `requests` exception (timeout, refused connection, DNS error, HTTP 4xx/5xx) triggers the
  offline path, so the kiosk does not crash on network problems.
* **One definition, three runtimes.** Models, preprocessing and the score → probability → label rule live in
  `common.py` only.
* **Thread-safety.** snnTorch layers keep membrane state inside the module, so the API serialises inference with a lock.
* **Note:** the online path (CNN by default) and the offline path (SNN) can use different models, so the two answers
  are not guaranteed to match.

---

## 9. Limitations and future work

* Evaluate on higher-resolution data (e.g. the 224 × 224 MedMNIST+ variants or full chest X-ray datasets) and add
  cross-dataset validation; address the false-alarm rate (threshold tuning, class weighting, calibration).
* Learn `β` and the threshold per layer; try other encodings (rate / latency coding) and fewer time steps to reduce
  latency.
* Quantise the weights and explore pruning for smaller footprints.
* Deploy on real edge hardware (Raspberry Pi / Jetson) and, if available, neuromorphic hardware or a simulator
  (Loihi / Lava, SpiNNaker) to measure real energy instead of operation counts.
* Add a kiosk UI (camera / file picker, result screen) and automated tests.

---

## 10. Team

* Team members:
  1. Vidhi Mittal - 23UCS217
  2. Gaurav Kumar - 23UCS196
  3. Puja Das - 23UCS002
  4. Reddy Eswar Anush - 23UCS173
* Mentor: Dr. Ashim Saha, Department of CSE, NIT Agartala

[PneumoniaMNIST / MedMNIST](https://medmnist.com/), [snnTorch](https://snntorch.readthedocs.io/),
[PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/).
