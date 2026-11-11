import { useCallback, useEffect, useRef, useState } from "react";
import { ScanLine } from "lucide-react";

const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MIN_LAPLACIAN_VARIANCE = 400;
const MAX_GLARE_RATIO = 0.02;

export default function CropPerspectiveStep({ imageSrc, onConfirm, onCancel }) {
  const canvasRef = useRef(null);
  const previewRef = useRef(null);
  const sourceDataRef = useRef(null);
  const processedDataRef = useRef(null);
  const [dragIndex, setDragIndex] = useState(null);
  const [grayscale, setGrayscale] = useState(true);
  const grayscaleRef = useRef(grayscale);
  const [quality, setQuality] = useState(null);
  const [loadingConfirm, setLoadingConfirm] = useState(false);
  const [dimensions, setDimensions] = useState(null);
  const [corners, setCorners] = useState([
    { x: 0.08, y: 0.08 }, { x: 0.92, y: 0.08 }, { x: 0.92, y: 0.92 }, { x: 0.08, y: 0.92 },
  ]);
  const cornersRef = useRef([
    { x: 0.08, y: 0.08 }, { x: 0.92, y: 0.08 }, { x: 0.92, y: 0.92 }, { x: 0.08, y: 0.92 },
  ]);

  useEffect(() => {
    const img = previewRef.current;
    if (!img) return undefined;
    sourceDataRef.current = null;
    processedDataRef.current = null;
    setDimensions(null);
    setQuality(null);
    const handleLoad = () => {
      const { naturalWidth, naturalHeight } = img;
      if (!naturalWidth || !naturalHeight) {
        setQuality({ ok: false, message: "Unable to load this image. Retake photo." });
        return;
      }
      const scale = Math.min(1, 900 / Math.max(naturalWidth, naturalHeight));
      const canvas = canvasRef.current;
      canvas.width = Math.round(naturalWidth * scale);
      canvas.height = Math.round(naturalHeight * scale);
      setDimensions({ width: canvas.width, height: canvas.height });
      cornersRef.current = [
        { x: 0.08, y: 0.08 }, { x: 0.92, y: 0.08 },
        { x: 0.92, y: 0.92 }, { x: 0.08, y: 0.92 },
      ];
      setCorners(cornersRef.current);
      const context = canvas.getContext("2d");
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
      sourceDataRef.current = context.getImageData(0, 0, canvas.width, canvas.height);
      drawPreview();
    };
    const handleError = () => setQuality({ ok: false, message: "Unable to load this image. Retake photo." });
    img.addEventListener("load", handleLoad);
    img.addEventListener("error", handleError);
    if (img.complete) {
      if (img.naturalWidth) handleLoad();
      else handleError();
    }
    return () => {
      img.removeEventListener("load", handleLoad);
      img.removeEventListener("error", handleError);
    };
  }, [imageSrc]);

  const onPointerDown = (i) => (e) => {
    e.preventDefault();
    setQuality(null);
    setDragIndex(i);
  };
  const onPointerMove = useCallback((e) => {
    if (dragIndex === null || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const nextCorner = {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
    cornersRef.current[dragIndex] = nextCorner;
    setCorners(cornersRef.current.map((corner, index) => index === dragIndex ? nextCorner : corner));
    drawPreview();
  }, [dragIndex]);
  const onPointerUp = () => {
    if (dragIndex === null) return;
    setDragIndex(null);
    window.requestAnimationFrame(() => {
      const processed = processedDataRef.current;
      if (processed && dimensions) evaluateQuality(processed.data, dimensions.width, dimensions.height);
    });
  };
  useEffect(() => {
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [onPointerMove]);

  function drawPreview() {
    const c = canvasRef.current;
    const source = sourceDataRef.current;
    if (!source || !c) return;
    const ctx = c.getContext("2d");
    const { width: w, height: h } = c;
    const quad = cornersRef.current.map((p) => [p.x * w, p.y * h]);
    let output;
    try {
      if (!isConvexQuad(quad)) throw new Error("The selected corners do not form a valid crop.");
      const inverse = invert3x3(fitQuad([[0, 0], [w, 0], [w, h], [0, h]], quad));
      if (!inverse) throw new Error("The selected corners do not form a valid crop.");
      const input = source.data;
      output = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const denominator = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2];
          const sx = (inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]) / denominator;
          const sy = (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]) / denominator;
          const outputIndex = (y * w + x) * 4;
          if (!Number.isFinite(sx) || !Number.isFinite(sy) || sx < 0 || sx >= w || sy < 0 || sy >= h) {
            output[outputIndex + 3] = 255;
            continue;
          }
          const x0 = Math.floor(sx), y0 = Math.floor(sy);
          const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
          const dx = sx - x0, dy = sy - y0;
          const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
          const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
          const topLeft = (1 - dx) * (1 - dy), topRight = dx * (1 - dy);
          const bottomLeft = (1 - dx) * dy, bottomRight = dx * dy;
          const red = input[i00] * topLeft + input[i10] * topRight + input[i01] * bottomLeft + input[i11] * bottomRight;
          const green = input[i00 + 1] * topLeft + input[i10 + 1] * topRight + input[i01 + 1] * bottomLeft + input[i11 + 1] * bottomRight;
          const blue = input[i00 + 2] * topLeft + input[i10 + 2] * topRight + input[i01 + 2] * bottomLeft + input[i11 + 2] * bottomRight;
          const gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
          output[outputIndex] = Math.round(grayscaleRef.current ? gray : red);
          output[outputIndex + 1] = Math.round(grayscaleRef.current ? gray : green);
          output[outputIndex + 2] = Math.round(grayscaleRef.current ? gray : blue);
          output[outputIndex + 3] = 255;
        }
      }
    } catch (renderError) {
      processedDataRef.current = null;
      setQuality({ ok: false, message: renderError.message || "Unable to correct this image. Retake photo." });
      ctx.clearRect(0, 0, w, h);
      return;
    }
    const imageData = ctx.createImageData(w, h);
    imageData.data.set(output);
    processedDataRef.current = imageData;
    ctx.putImageData(imageData, 0, 0);
    if (dragIndex === null) evaluateQuality(output, w, h);
  }

  function evaluateQuality(data, w, h) {
    const gray = new Uint8ClampedArray(w * h);
    let bright = 0;
    for (let i = 0, k = 0; i < data.length; i += 4, k++) {
      const value = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      gray[k] = value;
      if (value > 245) bright += 1;
    }
    const glareRatio = bright / gray.length;
    let sum = 0, sumSq = 0, n = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const lap = -gray[i - w - 1] - gray[i - w] - gray[i - w + 1] - gray[i - 1] + 8 * gray[i] - gray[i + 1] - gray[i + w - 1] - gray[i + w] - gray[i + w + 1];
        sum += lap; sumSq += lap * lap; n += 1;
      }
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const blurOk = variance >= MIN_LAPLACIAN_VARIANCE;
    const glareOk = glareRatio <= MAX_GLARE_RATIO;
    const ok = blurOk && glareOk;
    let message = "Ready to confirm";
    if (!blurOk) message = "Image looks blurry — hold steadier and retake photo.";
    else if (!glareOk) message = "Glare detected — adjust the film angle and retake photo.";
    setQuality({ ok, blurOk, glareOk, message, variance, glareRatio });
  }

  useEffect(() => {
    grayscaleRef.current = grayscale;
    if (sourceDataRef.current) drawPreview();
  }, [grayscale]);

  async function handleConfirm() {
    if (!quality?.ok || !processedDataRef.current) return;
    setLoadingConfirm(true);
    try {
      const outputCanvas = document.createElement("canvas");
      outputCanvas.width = dimensions.width;
      outputCanvas.height = dimensions.height;
      const outputContext = outputCanvas.getContext("2d");
      outputContext.putImageData(processedDataRef.current, 0, 0);
      const blob = await new Promise((resolve) => outputCanvas.toBlob(resolve, "image/png"));
      if (!blob || blob.size > MAX_FILE_BYTES) {
        setQuality((previous) => ({
          ...previous,
          ok: false,
          message: "Captured image is too large. Retake photo.",
        }));
        return;
      }
      onConfirm(new File([blob], "neuro-kiosk-capture.png", { type: "image/png" }));
    } catch (confirmError) {
      setQuality((previous) => ({
        ...previous,
        ok: false,
        message: confirmError.message || "Unable to save this image. Retake photo.",
      }));
    } finally {
      setLoadingConfirm(false);
    }
  }

  return (
    <section className="crop-step">
      <div className="crop-header"><h2>Review &amp; correct capture</h2><p className="crop-sub">Drag the corners to align the film edges, then confirm.</p></div>
      <div
        className="crop-canvas-wrap"
        style={dimensions ? { aspectRatio: `${dimensions.width} / ${dimensions.height}` } : undefined}
      >
        <canvas ref={canvasRef} className="crop-canvas" aria-label="Perspective-corrected preview" />
        {corners.map((corner, index) => (
          <button
            key={index}
            className="crop-handle"
            type="button"
            aria-label={`Move crop corner ${index + 1}`}
            style={{ left: `${corner.x * 100}%`, top: `${corner.y * 100}%` }}
            onPointerDown={onPointerDown(index)}
          />
        ))}
        <img ref={previewRef} src={imageSrc} alt="" className="visually-hidden" />
        <div className="quality-badge" role="status">
          <span className={`quality-dot ${quality?.ok ? "quality-ok" : "quality-bad"}`} />
          <span>{quality?.message ?? "Analyzing…"}</span>
        </div>
      </div>
      <div className="crop-controls">
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={grayscale}
            onChange={(event) => {
              grayscaleRef.current = event.target.checked;
              setGrayscale(event.target.checked);
            }}
          />
          <span>Auto-grayscale</span>
        </label>
        <button className="button button-outline" type="button" onClick={onCancel}>Retake photo</button>
        <button
          className="button button-primary"
          type="button"
          disabled={!quality?.ok || loadingConfirm}
          onClick={handleConfirm}
        >
          <ScanLine size={16} strokeWidth={1.6} /> Confirm capture
        </button>
      </div>
    </section>
  );
}

function fitQuad(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
    b.push(v);
  }
  const h = solve8(A, b);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}
function isConvexQuad(points) {
  let direction = 0;
  for (let i = 0; i < points.length; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % points.length];
    const [cx, cy] = points[(i + 2) % points.length];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (Math.abs(cross) < 1e-6) return false;
    const nextDirection = Math.sign(cross);
    if (direction && direction !== nextDirection) return false;
    direction = nextDirection;
  }
  return true;
}
function solve8(A, b) {
  const n = 8;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    }
    if (Math.abs(aug[piv][col]) < 1e-12) throw new Error("The selected corners do not form a valid crop.");
    [aug[col], aug[piv]] = [aug[piv], aug[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r][col] / aug[col][col];
      for (let c = col; c <= n; c++) aug[r][c] -= f * aug[col][c];
    }
  }
  return aug.map((row, i) => row[n] / row[i]);
}
function invert3x3(m) {
  const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-9) return null;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det],
  ];
}