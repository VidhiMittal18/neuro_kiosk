import { beforeEach, describe, expect, it, vi } from "vitest";
import { sendImageForDiagnosis } from "./api.js";
import { getPendingRecords, updateRecord } from "./db.js";
import { flushPending } from "./sync.js";

vi.mock("./api.js", () => ({
  sendImageForDiagnosis: vi.fn(),
}));
vi.mock("./db.js", () => ({
  getPendingRecords: vi.fn(),
  updateRecord: vi.fn(),
}));

describe("pending record sync", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "onLine", { configurable: true, value: true });
    updateRecord.mockResolvedValue({});
  });

  it("preserves JPEG type while sending a pending capture", async () => {
    getPendingRecords.mockResolvedValue([{
      id: 7,
      image: new Blob(["jpeg image"], { type: "image/jpeg" }),
    }]);
    sendImageForDiagnosis.mockResolvedValue({
      mode: "online",
      result: { prediction: "Normal" },
    });

    const outcome = await flushPending();

    const [file] = sendImageForDiagnosis.mock.calls[0];
    expect(file.type).toBe("image/jpeg");
    expect(file.name).toBe("neuro-kiosk-capture-7.jpg");
    expect(updateRecord).toHaveBeenCalledWith(7, expect.objectContaining({ status: "synced" }));
    expect(outcome).toMatchObject({ synced: 1, failed: 0 });
  });

  it("keeps a record pending when diagnosis still falls back offline", async () => {
    getPendingRecords.mockResolvedValue([{
      id: 8,
      image: new Blob(["png image"], { type: "image/png" }),
    }]);
    sendImageForDiagnosis.mockResolvedValue({
      mode: "offline",
      offline_reason: "Cloud API could not be reached.",
      result: { prediction: "Normal" },
    });

    const outcome = await flushPending();

    expect(updateRecord).toHaveBeenCalledWith(8, expect.objectContaining({
      status: "pending",
      mode: "offline",
      offlineReason: "Cloud API could not be reached.",
      syncedAt: null,
    }));
    expect(outcome).toMatchObject({ synced: 0, failed: 1 });
    expect(outcome.error).toBe("Cloud API could not be reached.");
  });
});
