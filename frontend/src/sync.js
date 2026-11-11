import { sendImageForDiagnosis } from "./api.js";
import { getPendingRecords, updateRecord } from "./db.js";

const SYNC_TAG = "sync-neuro-kiosk";

export function isOnline() {
  return typeof navigator !== "undefined" && navigator.onLine === true;
}

export async function registerBackgroundSync() {
  if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return false;
  if (typeof window === "undefined") return false;
  if (!("SyncManager" in window)) return false;
  try {
    const registration = await navigator.serviceWorker.ready;
    await registration.sync.register(SYNC_TAG);
    return true;
  } catch {
    return false;
  }
}

export async function flushPending() {
  if (!isOnline()) return { synced: 0, failed: 0, skipped: "offline" };
  const pending = await getPendingRecords();
  let synced = 0;
  let failed = 0;
  let lastError = null;
  for (const record of pending) {
    try {
      await syncRecord(record);
      synced += 1;
    } catch (error) {
      failed += 1;
      lastError = error;
    }
  }
  return { synced, failed, skipped: null, error: lastError?.message ?? null };
}

async function syncRecord(record) {
  const type = record.image.type || "image/png";
  const extension = type === "image/jpeg" ? "jpg" : "png";
  const file = new File([record.image], `neuro-kiosk-capture-${record.id}.${extension}`, {
    type,
  });
  const response = await sendImageForDiagnosis(file, false);
  await updateRecord(record.id, {
    status: response.mode === "online" ? "synced" : "pending",
    result: response,
    mode: response.mode,
    offlineReason: response.offline_reason ?? null,
    syncedAt: response.mode === "online" ? Date.now() : null,
  });
  if (response.mode !== "online") {
    throw new Error(response.offline_reason || "Cloud service is still unavailable.");
  }
}

export async function setupOnlineSync(onSynced) {
  if (typeof window === "undefined") return () => {};
  const handleOnline = async () => {
    if (!isOnline()) return;
    try {
      const result = await flushPending();
      if (onSynced) await onSynced(result);
    } catch (error) {
      if (onSynced) {
        await onSynced({
          synced: 0,
          failed: 1,
          skipped: null,
          error: error.message || "Unable to sync pending records.",
        });
      }
    }
  };
  window.addEventListener("online", handleOnline);
  return () => window.removeEventListener("online", handleOnline);
}