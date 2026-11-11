import { useEffect, useState } from "react";
import { Clock, ShieldAlert, ShieldCheck, ShieldOff, Trash2 } from "lucide-react";
import { deleteRecord, updateRecord } from "../db.js";

const STATUS_LABELS = { pending: "Pending", synced: "Synced", reviewed: "Reviewed" };
const STATUS_ICONS = { pending: ShieldOff, synced: ShieldCheck, reviewed: ShieldCheck };

function RecordThumbnail({ image }) {
  const [src, setSrc] = useState(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(image);
    setSrc(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [image]);

  return src ? <img src={src} alt="Captured image" className="thumb" /> : <span className="thumb-empty">Loading…</span>;
}

export default function ReviewQueue({ records, loading, onRecordsChange, onError }) {
  const [filter, setFilter] = useState("all");

  async function setStatus(id, status) {
    const updates = { status };
    if (status === "synced") updates.syncedAt = Date.now();
    if (status === "reviewed") updates.reviewedAt = Date.now();
    try {
      await updateRecord(id, updates);
      await onRecordsChange();
    } catch (updateError) {
      onError(updateError.message || "Unable to update this record.");
    }
  }

  async function toggleReviewed(id, current) {
    await setStatus(id, current === "reviewed" ? "synced" : "reviewed");
  }

  async function handleRemove(id) {
    if (!window.confirm("Remove this record from the queue?")) return;
    try {
      await deleteRecord(id);
      await onRecordsChange();
    } catch (deleteError) {
      onError(deleteError.message || "Unable to remove this record.");
    }
  }

  const filtered = records.filter((r) => filter === "all" || r.status === filter);

  return (
    <section className="review-queue" aria-labelledby="review-title">
      <div className="section-heading">
        <div className="section-number">02</div>
        <div>
          <p className="eyebrow">STORE-AND-FORWARD</p>
          <h2 id="review-title">Doctor review queue</h2>
        </div>
        <div className="status-counts" aria-label="Filter records">
          {["all", "pending", "synced", "reviewed"].map((status) => {
            const count = status === "all"
              ? records.length
              : records.filter((record) => record.status === status).length;
            const label = status === "all" ? "All" : STATUS_LABELS[status];
            return (
              <button
                key={status}
                className={`count ${filter === status ? "is-active" : ""}`}
                type="button"
                aria-pressed={filter === status}
                onClick={() => setFilter(status)}
              >
                {label} ({count})
              </button>
            );
          })}
        </div>
      </div>
      {loading ? (
        <p>Loading queue…</p>
      ) : filtered.length === 0 ? (
        <p className="empty-queue">No records yet. Capture or upload an image to begin.</p>
      ) : (
        <table className="review-table" aria-label="Review queue">
          <thead>
            <tr>
              <th>Capture</th>
              <th>Received</th>
              <th>Status</th>
              <th>Result</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((record) => {
              const result = record.result;
              const StatusIcon = STATUS_ICONS[record.status] || ShieldOff;
              const time = record.createdAt ? new Date(record.createdAt).toLocaleString() : "—";
              return (
                <tr key={record.id} className={`row status-${record.status}`}>
                  <td className="cell-thumb">
                    {record.image ? (
                      <RecordThumbnail image={record.image} />
                    ) : (
                      <span className="thumb-empty">—</span>
                    )}
                  </td>
                  <td className="cell-time"><Clock size={13} strokeWidth={1.5} /><span>{time}</span></td>
                  <td className="cell-status">
                    <span className={`status-badge status-${record.status}`}><StatusIcon size={13} strokeWidth={1.6} /> {STATUS_LABELS[record.status] ?? "Unknown"}</span>
                  </td>
                  <td className="cell-result">
                    {result ? <span>{result.result?.prediction ?? "—"}</span> : <span className="muted">—</span>}
                  </td>
                  <td className="cell-actions">
                    <button
                      className="icon-button"
                      type="button"
                      aria-label={record.status === "pending" ? "Pending synchronization" : `Mark ${record.status === "reviewed" ? "synced" : "reviewed"}`}
                      title={record.status === "pending" ? "This capture must sync before it can be reviewed." : undefined}
                      disabled={record.status === "pending"}
                      onClick={() => toggleReviewed(record.id, record.status)}
                    >
                      {record.status === "reviewed" ? <ShieldCheck size={14} strokeWidth={1.6} /> : <ShieldAlert size={14} strokeWidth={1.6} />}
                    </button>
                    <button className="icon-button" type="button" aria-label="Remove record" onClick={() => handleRemove(record.id)}><Trash2 size={14} strokeWidth={1.6} /></button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}