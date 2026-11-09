import { useEffect, useRef, useState } from "react";
import { ArrowUpFromLine, FileImage, LoaderCircle, RotateCcw, ScanLine } from "lucide-react";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const MAX_FILE_BYTES = 5 * 1024 * 1024;

export default function UploadPanel({ file, loading, onFileChange, onAnalyze, onSample }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [validationError, setValidationError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function acceptFile(nextFile) {
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setValidationError("Choose a PNG or JPEG image.");
      onFileChange(null);
      return;
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setValidationError("Image must be 5 MB or smaller.");
      onFileChange(null);
      return;
    }
    setValidationError("");
    onFileChange(nextFile);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function clearFile() {
    onFileChange(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section className="upload-section" aria-labelledby="upload-title">
      <div className="section-heading">
        <div className="section-number">01</div>
        <div><p className="eyebrow">IMAGE INPUT</p><h2 id="upload-title">Analyze an image</h2></div>
        {file && <button className="icon-button clear-button" type="button" onClick={clearFile} aria-label="Clear selected image" title="Clear image"><RotateCcw size={17} /></button>}
      </div>

      {!file ? (
        <div
          className={`drop-zone${dragging ? " is-dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/png,image/jpeg"
            aria-label="Choose a PNG or JPEG image"
            onChange={(event) => acceptFile(event.target.files?.[0])}
          />
          <button className="drop-target" type="button" onClick={() => inputRef.current?.click()}>
            <span className="upload-glyph"><ArrowUpFromLine size={23} strokeWidth={1.8} /></span>
            <span className="drop-title">Drop an image here</span>
            <span className="drop-subtitle">or <span className="text-action">browse files</span></span>
            <span className="file-spec">PNG or JPEG <i /> Up to 5 MB</span>
          </button>
        </div>
      ) : (
        <div className="preview-frame">
          <img src={previewUrl} alt={`Preview of ${file.name}`} />
          <div className="preview-caption"><FileImage size={16} /><span title={file.name}>{file.name}</span><small>{(file.size / 1024).toFixed(0)} KB</small></div>
        </div>
      )}
      {validationError && <p className="upload-validation" role="alert">{validationError}</p>}

      <div className="upload-actions">
        <button className="button button-primary analyze-button" type="button" disabled={!file || loading} onClick={onAnalyze}>
          {loading ? <><LoaderCircle className="spin" size={17} /> Analyzing</> : <><ScanLine size={17} /> Analyze image</>}
        </button>
        <div className="sample-actions" aria-label="Sample images">
          <span className="sample-label">TRY A SAMPLE</span>
          <button type="button" onClick={() => onSample("sample_normal.png")}>Normal</button>
          <button type="button" onClick={() => onSample("sample_pneumonia.png")}>Pneumonia</button>
        </div>
      </div>
    </section>
  );
}