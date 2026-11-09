import { useEffect, useRef, useState } from "react";
import { Activity, BarChart3, ShieldCheck, X } from "lucide-react";
import { diagnose, getSampleFile, getStatus } from "./api.js";
import BenchmarkTab from "./components/BenchmarkTab.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import Header from "./components/Header.jsx";
import ResultCard from "./components/ResultCard.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

const TABS = [
  { id: "diagnose", label: "Diagnose", icon: Activity },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3 },
];

function readAcknowledgement() {
  try {
    return window.localStorage.getItem("neuro-kiosk-acknowledged") === "true";
  } catch {
    return false;
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState("diagnose");
  const [status, setStatus] = useState(null);
  const [simulateOffline, setSimulateOffline] = useState(false);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [acknowledged, setAcknowledged] = useState(readAcknowledgement);
  const acknowledgementRef = useRef(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const nextStatus = await getStatus();
        if (active) setStatus(nextStatus);
      } catch {
        if (active) setStatus(null);
      }
    };
    poll();
    const intervalId = window.setInterval(poll, 5000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (acknowledged) return undefined;
    acknowledgementRef.current?.focus();
    function keepFocusInDialog(event) {
      if (event.key === "Tab") {
        event.preventDefault();
        acknowledgementRef.current?.focus();
      }
    }
    window.addEventListener("keydown", keepFocusInDialog);
    return () => window.removeEventListener("keydown", keepFocusInDialog);
  }, [acknowledged]);

  async function handleAnalyze() {
    if (!file || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await diagnose(file, simulateOffline));
    } catch (requestError) {
      setError(requestError.message || "Unable to analyze this image.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSample(filename) {
    setError("");
    setResult(null);
    try {
      setFile(await getSampleFile(filename));
    } catch (requestError) {
      setError(requestError.message || "Unable to load the sample image.");
    }
  }

  function acknowledge() {
    try {
      window.localStorage.setItem("neuro-kiosk-acknowledged", "true");
    } catch {
      // The acknowledgement remains valid for this session if storage is unavailable.
    }
    setAcknowledged(true);
  }

  return (
    <>
      <Header
        status={status}
        simulateOffline={simulateOffline}
        onSimulateOfflineChange={setSimulateOffline}
      />
      <Disclaimer />
      {error && (
        <div className="toast toast-error" role="alert">
          <span>{error}</span>
          <button className="icon-button" type="button" aria-label="Dismiss error" onClick={() => setError("")}><X size={17} /></button>
        </div>
      )}
      <main className="workspace">
        <div className="workspace-heading">
          <div>
            <p className="eyebrow">PNEUMONIA MNIST · EDGE INFERENCE STUDY</p>
            <h1>Research console</h1>
          </div>
          <div className="local-indicator"><ShieldCheck size={16} /> Local-first processing</div>
        </div>

        <nav className="view-tabs" aria-label="Kiosk views" role="tablist">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              id={`tab-${id}`}
              type="button"
              role="tab"
              aria-selected={activeTab === id}
              aria-controls={`panel-${id}`}
              className={`view-tab${activeTab === id ? " is-active" : ""}`}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={17} strokeWidth={1.8} />
              {label}
            </button>
          ))}
          <span className="tab-rule" aria-hidden="true" />
        </nav>

        {activeTab === "diagnose" ? (
          <section id="panel-diagnose" role="tabpanel" aria-labelledby="tab-diagnose" className="diagnose-grid">
            <div className="diagnose-main">
              <UploadPanel
                file={file}
                loading={loading}
                onFileChange={(nextFile) => {
                  setFile(nextFile);
                  setResult(null);
                  setError("");
                }}
                onAnalyze={handleAnalyze}
                onSample={handleSample}
              />
              {result && <ResultCard response={result} />}
            </div>
            <aside className="workflow-aside" aria-label="Study context">
              <div className="aside-index">01 <span>MODEL SCOPE</span></div>
              <h2>One image.<br />Two classes.</h2>
              <p>Model output is limited to Normal or Pneumonia using the PneumoniaMNIST research dataset.</p>
              <div className="aside-divider" />
              <div className="scope-row"><span>Input pipeline</span><strong>Gray · 28 × 28</strong></div>
              <div className="scope-row"><span>Inference route</span><strong>{simulateOffline ? "On-device SNN" : "Cloud → edge fallback"}</strong></div>
              <div className="aside-foot"><span className="pulse-dot" /> {status?.snn_loaded ? "Edge model ready" : "Edge model status pending"}</div>
            </aside>
          </section>
        ) : (
          <section id="panel-benchmarks" role="tabpanel" aria-labelledby="tab-benchmarks">
            <BenchmarkTab />
          </section>
        )}
      </main>

      {!acknowledged && (
        <div className="modal-backdrop" role="presentation">
          <section className="acknowledgement-modal" role="dialog" aria-modal="true" aria-labelledby="ack-title">
            <div className="modal-mark"><ShieldCheck size={21} /></div>
            <p className="eyebrow">BEFORE YOU CONTINUE</p>
            <h2 id="ack-title">Research demonstration only</h2>
            <p>This software is not a medical device and does not provide a diagnosis. Do not use its output to guide care.</p>
            <button ref={acknowledgementRef} className="button button-primary modal-acknowledge" type="button" onClick={acknowledge}>
              I understand
            </button>
          </section>
        </div>
      )}
    </>
  );
}