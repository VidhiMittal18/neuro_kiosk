import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { diagnose, getBenchmarks, getStatus } from "./api.js";

vi.mock("./api.js", () => ({
  diagnose: vi.fn(),
  getBenchmarks: vi.fn(),
  getSampleFile: vi.fn(),
  getStatus: vi.fn(),
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
    window.localStorage.clear();
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

  it("uploads an image and shows the returned model output", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    render(<App />);
    const image = new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" });

    fireEvent.change(screen.getByLabelText("Choose a PNG or JPEG image"), { target: { files: [image] } });
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));

    await waitFor(() => expect(diagnose).toHaveBeenCalledWith(image, false));
    expect(await screen.findByText("Inference result")).toBeInTheDocument();
    expect(screen.getByText("Cloud inference")).toBeInTheDocument();
  });

  it("shows a service-unreachable toast when the edge API is down", async () => {
    window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    diagnose.mockRejectedValue(new Error("Kiosk service unreachable"));
    render(<App />);
    const image = new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Choose a PNG or JPEG image"), { target: { files: [image] } });
    fireEvent.click(screen.getByRole("button", { name: /Analyze image/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Kiosk service unreachable");
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