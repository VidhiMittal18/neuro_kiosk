import { Activity, Clock3, Cpu, ShieldAlert, ShieldCheck } from "lucide-react";
import SnnDetails from "./SnnDetails.jsx";

export default function ResultCard({ response }) {
  const { result, mode, offline_reason: offlineReason } = response;
  const isPneumonia = result.prediction === "Pneumonia";
  const confidence = Math.max(0, Math.min(100, result.confidence * 100));
  const BadgeIcon = mode === "offline" ? Cpu : Activity;

  return (
    <section className="result-card" aria-labelledby="result-title" aria-live="polite">
      <div className="result-topline">
        <div className="result-heading"><span className="result-kicker"><span className="result-live-dot" /> MODEL OUTPUT</span><h2 id="result-title">Inference result</h2></div>
        <span className={`mode-badge ${mode === "offline" ? "mode-offline" : "mode-online"}`}><BadgeIcon size={14} />{mode === "offline" ? "Offline · on-device SNN" : "Cloud inference"}</span>
      </div>

      <div className="prediction-row">
        <div className={`prediction-symbol ${isPneumonia ? "prediction-alert" : "prediction-normal"}`}>
          {isPneumonia ? <ShieldAlert size={25} /> : <ShieldCheck size={25} />}
        </div>
        <div className="prediction-copy"><span>Predicted class</span><strong className={isPneumonia ? "text-red" : "text-green"}>{result.prediction}</strong></div>
        <div className="result-metric"><span><Clock3 size={14} /> Latency</span><strong>{Number(result.latency_ms).toFixed(1)}<small> ms</small></strong></div>
      </div>

      <div className="confidence-block">
        <div className="confidence-label"><span>Model confidence</span><strong>{confidence.toFixed(1)}%</strong></div>
        <div className="confidence-track" role="meter" aria-label="Model confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow={confidence}>
          <span style={{ width: `${confidence}%` }} />
        </div>
      </div>

      <div className="result-source"><span>Source</span><strong>{result.source}</strong></div>
      {offlineReason && <p className="offline-reason"><Cpu size={14} /> {offlineReason}</p>}
      {result.snn && <SnnDetails snn={result.snn} />}
      <p className="result-caveat">Research model output only · Not a clinical assessment</p>
    </section>
  );
}