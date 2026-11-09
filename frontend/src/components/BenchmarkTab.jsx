import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowDownToLine, Expand, Image as ImageIcon, LoaderCircle, X } from "lucide-react";
import { apiUrl, getBenchmarks } from "../api.js";
import MetricsTable from "./MetricsTable.jsx";

const CHART_COLORS = { cnn: "#2674a8", snn: "#d36a45", dense: "#83918c" };

function ChartFrame({ title, unit, children, className = "" }) {
  return (
    <section className={`chart-frame ${className}`}>
      <div className="chart-frame-heading"><div><p className="eyebrow">{unit}</p><h3>{title}</h3></div></div>
      <div className="benchmark-chart">{children}</div>
    </section>
  );
}

function Lightbox({ onClose }) {
  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="lightbox-backdrop" role="presentation" onClick={onClose}>
      <section className="lightbox-dialog" role="dialog" aria-modal="true" aria-label="Expanded benchmark figure" onClick={(event) => event.stopPropagation()}>
        <div className="lightbox-toolbar"><span>Benchmark figure</span><button className="icon-button" type="button" onClick={onClose} aria-label="Close expanded figure"><X size={20} /></button></div>
        <img src={apiUrl("/benchmark-image")} alt="Expanded CNN and SNN benchmark results" />
      </section>
    </div>
  );
}

export default function BenchmarkTab() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");
  const [imageExpanded, setImageExpanded] = useState(false);

  useEffect(() => {
    let active = true;
    getBenchmarks()
      .then((data) => { if (active) setMetrics(data); })
      .catch((requestError) => { if (active) setError(requestError.message); });
    return () => { active = false; };
  }, []);

  const qualityData = useMemo(() => metrics ? [
    { metric: "Accuracy", CNN: metrics.cnn.quality.accuracy_pct, SNN: metrics.snn.quality.accuracy_pct },
    { metric: "Macro F1", CNN: metrics.cnn.quality.f1_macro * 100, SNN: metrics.snn.quality.f1_macro * 100 },
  ] : [], [metrics]);
  const latencyData = useMemo(() => metrics ? [
    { model: "CNN", mean: metrics.cnn.latency_ms_per_image.mean, deviation: metrics.cnn.latency_ms_per_image.std },
    { model: "SNN", mean: metrics.snn.latency_ms_per_image.mean, deviation: metrics.snn.latency_ms_per_image.std },
  ] : [], [metrics]);
  const operationData = useMemo(() => metrics ? [
    { name: "CNN MACs", operations: metrics.cnn.compute.macs, fill: CHART_COLORS.cnn },
    { name: "SNN SOPs", operations: metrics.snn.compute.sops_per_inference, fill: CHART_COLORS.snn },
    { name: "SNN dense equivalent", operations: metrics.snn.compute.dense_equivalent_synaptic_ops, fill: CHART_COLORS.dense },
  ] : [], [metrics]);

  if (error) return <div className="benchmark-state error-banner" role="alert">{error}</div>;
  if (!metrics) return <div className="benchmark-state"><LoaderCircle className="spin" size={20} /> Loading benchmark data</div>;

  return (
    <div className="benchmark-view">
      <div className="benchmark-heading">
        <div><p className="eyebrow">PNEUMONIAMNIST · TEST SET</p><h2>Model benchmarks</h2><p>Measured on CPU · {metrics.config.test_samples} test samples · {metrics.config.latency_iterations} latency iterations</p></div>
        <div className="download-actions">
          <a className="button button-secondary" href={apiUrl("/benchmarks")} download="benchmark_metrics.json"><ArrowDownToLine size={16} /> JSON</a>
          <a className="button button-secondary" href={apiUrl("/benchmark-image")} download="benchmark_results.png"><ArrowDownToLine size={16} /> Figure</a>
        </div>
      </div>

      <MetricsTable metrics={metrics} />

      <div className="charts-grid">
        <ChartFrame title="Classification quality" unit="SCORE · %">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={qualityData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
              <CartesianGrid vertical={false} stroke="#e7ebe7" />
              <XAxis dataKey="metric" tickLine={false} axisLine={false} tick={{ fill: "#66716e", fontSize: 11 }} />
              <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: "#89918e", fontSize: 10 }} tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, ""]} />
              <Legend verticalAlign="top" height={32} />
              <Bar dataKey="CNN" fill={CHART_COLORS.cnn} radius={[3, 3, 0, 0]} maxBarSize={38} />
              <Bar dataKey="SNN" fill={CHART_COLORS.snn} radius={[3, 3, 0, 0]} maxBarSize={38} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>

        <ChartFrame title="CPU inference latency" unit="MILLISECONDS · LOWER IS BETTER">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={latencyData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
              <CartesianGrid vertical={false} stroke="#e7ebe7" />
              <XAxis dataKey="model" tickLine={false} axisLine={false} tick={{ fill: "#66716e", fontSize: 11 }} />
              <YAxis tickLine={false} axisLine={false} tick={{ fill: "#89918e", fontSize: 10 }} />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(2)} ms`, "Mean"]} />
              <Bar dataKey="mean" fill={CHART_COLORS.cnn} radius={[3, 3, 0, 0]} maxBarSize={48}>
                <ErrorBar dataKey="deviation" width={5} stroke="#263b3b" direction="y" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>

        <ChartFrame title="Estimated operations per image" unit="OPERATIONS · LOG SCALE" className="chart-wide">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={operationData} layout="vertical" margin={{ top: 4, right: 24, bottom: 0, left: 36 }}>
              <CartesianGrid horizontal={false} stroke="#e7ebe7" />
              <XAxis type="number" scale="log" domain={[100000, "auto"]} tickLine={false} axisLine={false} tick={{ fill: "#89918e", fontSize: 10 }} tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`} />
              <YAxis type="category" dataKey="name" width={152} tickLine={false} axisLine={false} tick={{ fill: "#66716e", fontSize: 11 }} />
              <Tooltip formatter={(value) => [new Intl.NumberFormat("en-US").format(value), "Estimated operations"]} />
              <Bar dataKey="operations" radius={[0, 3, 3, 0]} maxBarSize={26}>
                {operationData.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      </div>

      <section className="benchmark-explainer">
        <div className="explainer-mark">i</div>
        <div><h3>Reading the SNN numbers</h3><p>SOPs count synaptic operations triggered by spikes; spike sparsity is the share of neuron-steps that emitted a spike. These are operation estimates, not direct energy measurements. In this CPU benchmark the SNN is slower than the CNN; its latency is not an advantage here.</p></div>
      </section>

      <section className="figure-section" aria-labelledby="figure-title">
        <div className="figure-heading"><div><p className="eyebrow">SOURCE FIGURE</p><h3 id="figure-title">Benchmark results</h3></div><button className="icon-button" type="button" onClick={() => setImageExpanded(true)} aria-label="Expand benchmark figure" title="Expand figure"><Expand size={18} /></button></div>
        <button className="figure-preview" type="button" onClick={() => setImageExpanded(true)} aria-label="Open expanded benchmark figure">
          <img src={apiUrl("/benchmark-image")} alt="Charts showing CNN and SNN test accuracy, CPU latency, compute, memory, and spike activity" />
          <span><ImageIcon size={15} /> Click to expand</span>
        </button>
      </section>

      {imageExpanded && <Lightbox onClose={() => setImageExpanded(false)} />}
    </div>
  );
}