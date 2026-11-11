import { useEffect, useRef, useState } from "react";
import { Camera, ScanLine, X } from "lucide-react";
import CropPerspectiveStep from "./CropPerspectiveStep.jsx";


export default function CameraPanel({ onFileReady, loading }) {
  const [cameraActive, setCameraActive] = useState(false);
  const [step, setStep] = useState("instructions");
  const [capturedSrc, setCapturedSrc] = useState(null);
  const [cameraError, setCameraError] = useState("");
  const [videoReady, setVideoReady] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    if (step === "viewfinder" && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [step]);

  useEffect(() => () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    if (capturedSrc) URL.revokeObjectURL(capturedSrc);
  }, [capturedSrc]);

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      streamRef.current = stream;
      setCameraActive(true);
      setVideoReady(false);
      setStep("viewfinder");
    } catch {
      setCameraError("Camera access unavailable — use Upload instead.");
    }
  }

  function clearCapturedPhoto() {
    if (capturedSrc) URL.revokeObjectURL(capturedSrc);
    setCapturedSrc(null);
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setVideoReady(false);
    setStep("instructions");
    clearCapturedPhoto();
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setCameraError("Camera is still starting — try again in a moment.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setCameraError("Unable to capture an image — use Upload instead.");
      return;
    }
    context.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("Unable to capture an image — try again or use Upload.");
        return;
      }
      const url = URL.createObjectURL(blob);
      setCapturedSrc(url);
      setCameraError("");
      setStep("review");
    }, "image/png");
  }

  function retake() {
    clearCapturedPhoto();
    setStep("viewfinder");
  }

  function handleFileReady(file) {
    clearCapturedPhoto();
    stopCamera();
    setStep("instructions");
    onFileReady(file);
  }

  return (
    <section className="camera-panel" aria-labelledby="camera-title">
      <div className="section-heading">
        <div className="section-number">01</div>
        <div>
          <p className="eyebrow">IMAGE INPUT</p>
          <h2 id="camera-title">Capture or upload</h2>
        </div>
      </div>
      {step === "instructions" && (
        <div className="camera-instructions">
          <h3>How to scan a film</h3>
          <ol>
            <li>Place the X-ray film on a light box or bright window.</li>
            <li>Position the camera directly above, parallel to the film.</li>
            <li>Center the film inside the on-screen alignment box.</li>
            <li>Avoid glare — tilt slightly if reflections appear.</li>
            <li>Tap <strong>Capture photo</strong>, then confirm the crop.</li>
          </ol>
          {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}
          <div className="camera-actions">
            {cameraActive ? (
              <button className="button button-outline" type="button" onClick={stopCamera}>
                <X size={16} strokeWidth={1.7} /> Stop camera
              </button>
            ) : (
              <button className="button button-primary" type="button" onClick={startCamera} disabled={loading}>
                <Camera size={16} strokeWidth={1.7} /> Start camera
              </button>
            )}
          </div>
        </div>
      )}

      {step === "viewfinder" && cameraActive && (
        <div className="viewfinder">
          <div className="video-wrap">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
              onLoadedMetadata={() => setVideoReady(true)}
            />
            <div className="camera-guide" aria-hidden="true">
              <div className="guide-frame">
                <div className="corner tl" />
                <div className="corner tr" />
                <div className="corner bl" />
                <div className="corner br" />
                <div className="guide-center" />
              </div>
              <p className="guide-hint">Place the X-ray film centered inside the box, flush against the light box or window.</p>
            </div>
            <div className="capture-layer">
              <button className="button capture-button" type="button" onClick={capturePhoto} disabled={loading || !videoReady}>
                <ScanLine size={20} strokeWidth={1.7} /> Capture photo
              </button>
            </div>
          </div>
          <p className="viewfinder-hint">Center the film in the box, then capture.</p>
        </div>
      )}

      {step === "review" && capturedSrc && (
        <CropPerspectiveStep imageSrc={capturedSrc} onConfirm={handleFileReady} onCancel={retake} />
      )}
    </section>
  );
}