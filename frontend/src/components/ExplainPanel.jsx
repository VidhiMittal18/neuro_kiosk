import { useEffect, useState } from "react";
import { explainImage } from "../api.js";
import "./ExplainPanel.css";

const METHODS = [
  { value: "grad2d", label: "SNN-Grad2D" },
  { value: "grad3d", label: "SNN-Grad3D" },
  { value: "ig2d", label: "SNN-IG2D" },
  { value: "ig3d", label: "SNN-IG3D" },
  { value: "sam", label: "SAM" },
  { value: "inverted_sam", label: "Inverted SAM" },
  { value: "cam", label: "SNN-CAM" },
];

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(6)
    : "Unavailable";
}

export default function ExplainPanel({ file }) {
  const [method, setMethod] = useState("grad2d");
  const [timesteps, setTimesteps] = useState(16);
  const [igSteps, setIgSteps] = useState(50);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [originalUrl, setOriginalUrl] = useState("");

  useEffect(() => {
    if (!file) {
      setOriginalUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setOriginalUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  useEffect(() => {
    setExplanation(null);
    setError("");
  }, [file]);

  async function handleExplain(event) {
    event.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setExplanation(null);

    try {
      const result = await explainImage(file, {
        method,
        timesteps,
        targetClass: "predicted",
        igSteps,
      });
      setExplanation(result);
    } catch (requestError) {
      setError(requestError.message || "Unable to generate the explanation.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="xai-panel" aria-labelledby="xai-title">
      <div className="xai-panel-heading">
        <div>
          <p className="xai-eyebrow">EXPLAINABLE AI</p>
          <h2 id="xai-title">Explain with DenseNet121 SNN</h2>
          <p className="xai-intro">
            Generates an attribution view using the converted SpikingJelly SNN.
            This is a separate research model and may not explain the cloud ANN
            or the existing kiosk fallback prediction.
          </p>
        </div>
      </div>

      <form className="xai-controls" onSubmit={handleExplain}>
        <label>
          Explanation method
          <select value={method} onChange={(event) => setMethod(event.target.value)}>
            {METHODS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>

        <label>
          SNN timesteps
          <select
            value={timesteps}
            onChange={(event) => setTimesteps(Number(event.target.value))}
          >
            <option value={16}>16</option>
          </select>
        </label>

        {(method === "ig2d" || method === "ig3d") && (
          <label>
            Integration steps
            <input
              type="number"
              min="10"
              max="100"
              step="5"
              value={igSteps}
              onChange={(event) => setIgSteps(Number(event.target.value))}
            />
          </label>
        )}

        <button className="button button-primary xai-submit" type="submit" disabled={!file || loading}>
          {loading ? "Generating explanation…" : "Generate explanation"}
        </button>
      </form>

      {loading && (
        <p className="xai-status" role="status">
          Running SNN explanation on the CPU. Integrated Gradients may take longer.
        </p>
      )}

      {error && <p className="xai-error" role="alert">{error}</p>}

      {explanation && (
        <div className="xai-results">
          <div className="xai-result-summary">
            <div><span>Explaining model</span><strong>DenseNet121 SpikingJelly SNN</strong></div>
            <div><span>SNN prediction</span><strong>{explanation.prediction}</strong></div>
            <div><span>Explanation target</span><strong>{explanation.target_class}</strong></div>
            <div><span>Timesteps</span><strong>{explanation.timesteps}</strong></div>
            <div><span>Raw pneumonia score</span><strong>{formatScore(explanation.pneumonia_score)}</strong></div>
            <div><span>Decision threshold</span><strong>{formatScore(explanation.threshold)}</strong></div>
          </div>

          <div className="xai-image-grid">
            {originalUrl && (
              <figure>
                <img src={originalUrl} alt="Original uploaded chest X-ray" />
                <figcaption>Original X-ray</figcaption>
              </figure>
            )}
            {explanation.heatmap_data_url && (
              <figure>
                <img src={explanation.heatmap_data_url} alt={`${explanation.method} attribution heatmap`} />
                <figcaption>{explanation.method?.toUpperCase()} heatmap</figcaption>
              </figure>
            )}
            {explanation.overlay_data_url && (
              <figure>
                <img src={explanation.overlay_data_url} alt={`${explanation.method} heatmap overlaid on the X-ray`} />
                <figcaption>Attribution overlay</figcaption>
              </figure>
            )}
            {explanation.temporal_data_url && (
              <figure className="xai-temporal-figure">
                <img src={explanation.temporal_data_url} alt={`${explanation.method} visualization across SNN timesteps`} />
                <figcaption>Temporal explanation across timesteps</figcaption>
              </figure>
            )}
          </div>

          <p className="xai-caveat">
            Attribution visualizations show which inputs or internal features
            influenced this model output. They are not verified pneumonia
            lesion maps and are not for clinical decision-making.
          </p>
        </div>
      )}
    </section>
  );
}
