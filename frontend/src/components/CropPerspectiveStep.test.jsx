import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CropPerspectiveStep from "./CropPerspectiveStep.jsx";

function createCanvasContext() {
  const width = 96;
  const height = 96;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = (Math.floor(x / 6) + Math.floor(y / 6)) % 2 ? 180 : 60;
      const index = (y * width + x) * 4;
      data[index] = value;
      data[index + 1] = value;
      data[index + 2] = value;
      data[index + 3] = 255;
    }
  }
  return {
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data })),
    createImageData: vi.fn((imageWidth, imageHeight) => ({
      data: new Uint8ClampedArray(imageWidth * imageHeight * 4),
    })),
    putImageData: vi.fn(),
    clearRect: vi.fn(),
  };
}

describe("CropPerspectiveStep", () => {
  it("waits for the image, renders movable corners, and exports the corrected image", async () => {
    const context = createCanvasContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((callback) => {
      callback(new Blob(["corrected image"], { type: "image/png" }));
    });
    const onConfirm = vi.fn();
    const { container } = render(
      <CropPerspectiveStep imageSrc="blob:test-image" onConfirm={onConfirm} onCancel={vi.fn()} />
    );
    const image = container.querySelector("img");
    Object.defineProperties(image, {
      complete: { configurable: true, value: false },
      naturalWidth: { configurable: true, value: 96 },
      naturalHeight: { configurable: true, value: 96 },
    });

    expect(screen.getByRole("button", { name: "Confirm capture" })).toBeDisabled();
    fireEvent.load(image);

    await waitFor(() => expect(screen.getByRole("button", { name: "Confirm capture" })).toBeEnabled());
    const corners = screen.getAllByRole("button", { name: /Move crop corner/ });
    expect(corners).toHaveLength(4);
    vi.spyOn(container.querySelector("canvas"), "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 96,
      height: 96,
      right: 96,
      bottom: 96,
    });
    fireEvent.pointerDown(corners[0], { clientX: 8, clientY: 8 });
    const move = new Event("pointermove");
    Object.defineProperties(move, {
      clientX: { value: 16 },
      clientY: { value: 16 },
    });
    fireEvent(window, move);
    expect(corners[0].style.left).toBe("16.666666666666664%");
    fireEvent.pointerUp(window);
    await waitFor(() => expect(screen.getByRole("button", { name: "Confirm capture" })).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: "Confirm capture" }));

    await waitFor(() => expect(onConfirm).toHaveBeenCalledOnce());
    expect(onConfirm.mock.calls[0][0]).toBeInstanceOf(File);
    expect(onConfirm.mock.calls[0][0].type).toBe("image/png");
  });
});
