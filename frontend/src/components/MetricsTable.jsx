const percent = (value) => `${(value * 100).toFixed(1)}%`;
const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export default function MetricsTable({ metrics }) {
  const cnn = metrics.cnn;
  const snn = metrics.snn;
  const rows = [
    ["Accuracy", `${cnn.quality.accuracy_pct.toFixed(1)}%`, `${snn.quality.accuracy_pct.toFixed(1)}%`, "Higher is better"],
    ["Macro F1", percent(cnn.quality.f1_macro), percent(snn.quality.f1_macro), "Higher is better"],
    ["Macro precision", percent(cnn.quality.precision_macro), percent(snn.quality.precision_macro), "Higher is better"],
    ["Macro recall", percent(cnn.quality.recall_macro), percent(snn.quality.recall_macro), "Higher is better"],
    ["CPU latency / image", `${cnn.latency_ms_per_image.mean.toFixed(2)} ± ${cnn.latency_ms_per_image.std.toFixed(2)} ms`, `${snn.latency_ms_per_image.mean.toFixed(2)} ± ${snn.latency_ms_per_image.std.toFixed(2)} ms`, "Measured on CPU"],
    ["Trainable parameters", integer.format(cnn.footprint.trainable_params), integer.format(snn.footprint.trainable_params), "Model size"],
    ["Model memory", `${cnn.footprint.memory_mb.toFixed(3)} MB`, `${snn.footprint.memory_mb.toFixed(3)} MB`, "Weights only"],
    ["Compute / image", `${integer.format(cnn.compute.macs)} MACs`, `${integer.format(snn.compute.sops_per_inference)} SOPs`, "Different operation types"],
    ["Mean spike rate", "—", percent(snn.sparsity.mean_spike_rate_per_neuron_per_step), "SNN activity"],
    ["SOPs / dense equivalent", "—", percent(snn.compute.sops_to_dense_ratio), "SNN estimate"],
  ];

  return (
    <div className="metrics-table-wrap">
      <table className="metrics-table">
        <caption className="visually-hidden">CNN and SNN benchmark comparison</caption>
        <thead><tr><th scope="col">Metric</th><th scope="col">CNN</th><th scope="col">SNN</th><th scope="col">Context</th></tr></thead>
        <tbody>
          {rows.map(([label, cnnValue, snnValue, context]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{cnnValue}</td>
              <td>{snnValue}</td>
              <td className="metric-context">{context}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}