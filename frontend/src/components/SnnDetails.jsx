import { Activity, Zap } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export default function SnnDetails({ snn }) {
  const rates = Object.entries(snn.per_layer_spike_rate || {}).map(([layer, rate]) => ({
    layer: layer.toUpperCase(),
    rate: rate * 100,
  }));

  return (
    <section className="snn-panel" aria-labelledby="snn-title">
      <div className="snn-heading"><div><p className="eyebrow">LOCAL MODEL TELEMETRY</p><h3 id="snn-title">Spiking network activity</h3></div><span className="snn-step-count">{snn.time_steps} steps</span></div>
      <div className="snn-metrics">
        <div><span><Activity size={13} /> Total spikes</span><strong>{numberFormat.format(snn.spikes_this_inference)}</strong></div>
        <div><span><Zap size={13} /> SOPs / inference</span><strong>{numberFormat.format(snn.sops_this_inference)}</strong></div>
        <div><span>Dense equivalent</span><strong>{numberFormat.format(snn.dense_equivalent_ops)}</strong></div>
        <div><span>Mean spike rate</span><strong>{(snn.mean_spike_rate_per_neuron_per_step * 100).toFixed(1)}%</strong></div>
      </div>
      <div className="spike-counts">
        {Object.entries(snn.output_spike_counts || {}).map(([label, count]) => (
          <div key={label}><span>{label} output spikes</span><strong>{count}</strong></div>
        ))}
      </div>
      <div className="layer-chart" aria-label="Per-layer spike rate chart">
        <span className="chart-caption">SPIKE RATE BY LAYER</span>
        <div className="layer-chart-height">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rates} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
              <CartesianGrid vertical={false} stroke="#e7ebe7" />
              <XAxis dataKey="layer" tickLine={false} axisLine={false} tick={{ fill: "#67716e", fontSize: 11 }} />
              <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: "#89918e", fontSize: 10 }} tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, "Spike rate"]} cursor={{ fill: "#f1f5f2" }} />
              <Bar dataKey="rate" fill="#147d70" radius={[3, 3, 0, 0]} maxBarSize={38} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}