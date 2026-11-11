const DB_NAME = "neuro-kiosk";
const DB_VERSION = 1;
const STORE_NAME = "records";
const SYNC_TAG = "sync-neuro-kiosk";
const API_BASE = new URL(self.location.href).searchParams.get("apiBase") === "/api" ? "/api" : "";
const API_PATH = `${API_BASE}/diagnose`;

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
        store.createIndex("status", "status", { unique: false });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("IndexedDB open was blocked"));
  });
}

function getPendingRecords(db) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index("status");
    const request = index.openCursor(IDBKeyRange.only("pending"));
    const rows = [];
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        rows.push(cursor.value);
        cursor.continue();
      } else {
        resolve(rows);
      }
    };
    request.onerror = () => reject(request.error);
  });
}

function updateRecordInDb(db, record, json) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const getRequest = store.get(record.id);
    getRequest.onsuccess = () => {
      const current = getRequest.result;
      if (current) {
        const synced = json.mode === "online";
        const updated = {
          ...current,
          status: synced ? "synced" : "pending",
          result: json,
          mode: json.mode,
          offlineReason: json.offline_reason ?? null,
          syncedAt: synced ? Date.now() : null,
          updatedAt: Date.now(),
        };
        store.put(updated);
      }
    };
    getRequest.onerror = () => reject(getRequest.error);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function sendRecord(db, record) {
  const type = record.image.type || "image/png";
  const extension = type === "image/jpeg" ? "jpg" : "png";
  const form = new FormData();
  form.append("file", new File([record.image], `capture-${record.id}.${extension}`, { type }));
  form.append("simulate_offline", "false");
  const response = await fetch(API_PATH, { method: "POST", body: form });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const json = await response.json();
  await updateRecordInDb(db, record, json);
  if (json.mode !== "online") {
    throw new Error(json.offline_reason || "Cloud service is still unavailable.");
  }
}

self.addEventListener("sync", (event) => {
  if (event.tag !== SYNC_TAG) return;
  event.waitUntil(
    (async () => {
      const db = await openDb();
      try {
        const pending = await getPendingRecords(db);
        let failed = false;
        for (const record of pending) {
          try {
            await sendRecord(db, record);
          } catch {
            failed = true;
          }
        }
        if (failed) throw new Error("Background sync failed; pending records remain queued.");
      } finally {
        db.close();
      }
    })()
  );
});