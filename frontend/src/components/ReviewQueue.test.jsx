import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ReviewQueue from "./ReviewQueue.jsx";
import { updateRecord } from "../db.js";

vi.mock("../db.js", () => ({
  deleteRecord: vi.fn(),
  updateRecord: vi.fn(),
}));

describe("ReviewQueue", () => {
  const records = [
    { id: 1, status: "pending", createdAt: Date.now() },
    { id: 2, status: "synced", createdAt: Date.now() },
  ];

  beforeEach(() => {
    updateRecord.mockResolvedValue({});
  });

  it("keeps pending records in the sync queue until they have synced", () => {
    render(
      <ReviewQueue
        records={records}
        loading={false}
        onRecordsChange={vi.fn()}
        onError={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Pending synchronization" })).toBeDisabled();
  });

  it("allows a synced record to be marked reviewed", async () => {
    const onRecordsChange = vi.fn();
    render(
      <ReviewQueue
        records={records}
        loading={false}
        onRecordsChange={onRecordsChange}
        onError={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Mark reviewed" }));

    await waitFor(() => {
      expect(updateRecord).toHaveBeenCalledWith(2, expect.objectContaining({ status: "reviewed" }));
      expect(onRecordsChange).toHaveBeenCalled();
    });
  });
});
