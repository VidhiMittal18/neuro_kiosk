import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { diagnose, getBenchmarks, getSampleFile, getStatus } from "./api.js";
import { addRecord, getAllRecords, updateRecord } from "./db.js";
import { registerBackgroundSync } from "./sync.js";

vi.mock("./api.js", () => ({
  apiUrl: vi.fn((path) => path),
  diagnose: vi.fn(),
  getBenchmarks: vi.fn(),
  getSampleFile: vi.fn(),
  getStatus: vi.fn(),
}));
vi.mock("./db.js", () => ({
  addRecord: vi.fn().mockResolvedValue(1),
  updateRecord: vi.fn().mockResolvedValue({}),
  getAllRecords: vi.fn().mockResolvedValue([]),
}));
vi.mock("./sync.js", () => ({
  registerBackgroundSync: vi.fn().mockResolvedValue(true),
  setupOnlineSync: vi.fn().mockResolvedValue(() => {}),
  flushPending: vi.fn().mockResolvedValue({ synced: 0, failed: 0, skipped: null }),
  isOnline: vi.fn(() => true),
}));
vi.mock("./components/CameraPanel.jsx", () => ({
  default: function MockCameraPanel({ onFileReady }) {
    return (
      <div data-testid="camera-panel">
        <p>Place the film centered inside the alignment box.</p>
        <button type="button" onClick={() => onFileReady(new File([new Uint8Array([1])], "capture.png", { type: "image/png" }))}>
          Capture photo
        </button>
      </div>
    );
  },
}));
vi.mock("./components/ReviewQueue.jsx", () => ({
  default: function MockReviewQueue({ records }) {
    return (
      <div data-testid="review-queue">
        {records.map((r) => (
          <div key={r.id} data-testid="review-row" data-status={r.status}>
            {r.id}: {r.status}
          </div>
        ))}
      </div>
    );
  },
}));

const benchmarkFixture = {
  config: { dataset: "PneumoniaMNIST", test_samples: 624, latency_iterations: 100 },
  cnn: {
    quality: { accuracy_pct: 86.2, f1_macro: 0.838, precision_macro: 0.904, recall_macro: 0.818 },
    latency_ms_per_image: { mean: 0.4, std: 0.18 },
    footprint: { trainable_params: 8034, memory_mb: 0.031 },
    compute: { macs: 1019200 },
  },
  snn: {
    quality: { accuracy_pct: 86.1, f1_macro: 0.839, precision_macro: 0.89, recall_macro: 0.821 },
    latency_ms_per_image: { mean: 24, std: 5.7 },
    footprint: { trainable_params: 10970, memory_mb: 0.05 },
    compute: { sops_per_inference: 1703640, dense_equivalent_synaptic_ops: 9231360, sops_to_dense_ratio: 0.1845 },
    sparsity: { mean_spike_rate_per_neuron_per_step: 0.23 },
  },
};

describe("Neuro Kiosk", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    getAllRecords.mockResolvedValue([]);
    getStatus.mockResolvedValue({ cloud_reachable: true, snn_loaded: true });
    getBenchmarks.mockResolvedValue(benchmarkFixture);
    diagnose.mockResolvedValue({
      mode: "online",
      offline_reason: null,
      result: { prediction: "Normal", confidence: 0.91, latency_ms: 4.2, source: "Cloud-API" },
    });
  });

  it("requires acknowledgement and keeps the disclaimer visible afterward", () => {
    render(<App />);

    expect(screen.getByRole("dialog", { name: "Research demonstration only" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "I understand" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByText(/Not for clinical use/)).toBeInTheDocument();
    expect(window.localStorage.getItem("neuro-kiosk-acknowledged")).toBe("true");
  });

  function switchToUpload() {
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
  }

  it("renders camera input with scanning instructions", () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);

    expect(screen.getByTestId("camera-panel")).toBeInTheDocument();
    expect(screen.getByText(/place the film centered inside the alignment box/i)).toBeInTheDocument();
  });

  it("switches between camera and upload input modes", () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);

    expect(screen.getByTestId("camera-panel")).toBeInTheDocument();
    switchToUpload();
    expect(screen.getByLabelText("Choose a PNG or JPEG image")).toBeInTheDocument();
  });

  it("uploads an image and shows the returned model output", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);
    switchToUpload();
    const image = new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" });

    fireEvent.change(screen.getByLabelText("Choose a PNG or JPEG image"), { target: { files: [image] } });
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));

    await waitFor(() => expect(addRecord).toHaveBeenCalledWith(
      expect.objectContaining({ status: "pending", image })
    ));
    await waitFor(() => expect(diagnose).toHaveBeenCalledWith(image, false));
    await waitFor(() => expect(updateRecord).toHaveBeenCalledWith(1, expect.objectContaining({ status: "synced" })));
    expect(await screen.findByText("Inference result")).toBeInTheDocument();
    expect(screen.getByText("Cloud inference")).toBeInTheDocument();
  });

  it("saves a capture as pending and marks it synced when online", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);
    fireEvent.click(screen.getByTestId("camera-panel").querySelector("button"));
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));
    await waitFor(() => expect(addRecord).toHaveBeenCalledWith(
      expect.objectContaining({ status: "pending" })
    ));
    await waitFor(() => expect(updateRecord).toHaveBeenCalledWith(1, expect.objectContaining({ status: "synced" })));
  });

  it("keeps a record pending when the diagnosis falls back to offline", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    diagnose.mockResolvedValueOnce({
      mode: "offline",
      offline_reason: "cloud API could not be reached.",
      result: { prediction: "Normal", confidence: 0.5, latency_ms: 24, source: "SNN" },
    });
    render(<App />);
    switchToUpload();
    const image = new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Choose a PNG or JPEG image"), { target: { files: [image] } });
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));

    await waitFor(() => expect(addRecord).toHaveBeenCalledWith(
      expect.objectContaining({ status: "pending" })
    ));
    await waitFor(() => expect(updateRecord).toHaveBeenCalledWith(1, expect.objectContaining({ status: "pending" })));
    expect(registerBackgroundSync).toHaveBeenCalled();
  });

  it("shows a service-unreachable toast when the edge API is down", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    diagnose.mockRejectedValue(new Error("Kiosk service unreachable"));
    render(<App />);
    switchToUpload();
    const image = new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Choose a PNG or JPEG image"), { target: { files: [image] } });
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Kiosk service unreachable");
    await waitFor(() => expect(addRecord).toHaveBeenCalledWith(
      expect.objectContaining({ status: "pending", image })
    ));
    expect(registerBackgroundSync).toHaveBeenCalled();
  });

  it("shows the review queue on its tab", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    getAllRecords.mockResolvedValue([{ id: 1, status: "pending", createdAt: Date.now() }]);
    render(<App />);

    fireEvent.click(screen.getByRole("tab", { name: "Review queue" }));
    await waitFor(() => expect(screen.getByTestId("review-queue")).toBeInTheDocument());
    expect(screen.getByTestId("review-row")).toHaveAttribute("data-status", "pending");
  });

  it("loads the benchmark comparison when its tab is opened", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Benchmarks" }));

    expect(await screen.findByRole("heading", { name: "Model benchmarks" })).toBeInTheDocument();
    expect(await screen.findByRole("table", { name: "CNN and SNN benchmark comparison" })).toBeInTheDocument();
    expect(screen.getByText(/SNN is slower than the CNN/)).toBeInTheDocument();
  });
});