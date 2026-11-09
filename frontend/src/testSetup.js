import "@testing-library/jest-dom/vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock;
globalThis.URL.createObjectURL ??= () => "blob:neuro-kiosk-test";
globalThis.URL.revokeObjectURL ??= () => {};