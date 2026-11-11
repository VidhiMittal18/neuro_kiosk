const DB_NAME = "neuro-kiosk";
const DB_VERSION = 1;
const STORE_NAME = "records";

let cachedDb = null;

function openDb() {
  if (cachedDb) return Promise.resolve(cachedDb);
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
    request.onsuccess = () => {
      cachedDb = request.result;
      cachedDb.onclose = () => { cachedDb = null; };
      resolve(cachedDb);
    };
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("IndexedDB open was blocked"));
  });
}

export async function addRecord(record) {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  const now = Date.now();
  const entry = {
    ...record,
    createdAt: record.createdAt ?? now,
    updatedAt: now,
    status: record.status ?? "pending",
    syncedAt: record.syncedAt ?? null,
    reviewedAt: record.reviewedAt ?? null,
    reviewNotes: record.reviewNotes ?? null,
  };
  return new Promise((resolve, reject) => {
    const req = store.add(entry);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getAllRecords() {
  return queryByIndex("createdAt", null, "prev");
}

export async function getPendingRecords() {
  return queryByIndex("status", IDBKeyRange.only("pending"), "next").then((rows) =>
    rows.sort((a, b) => b.createdAt - a.createdAt)
  );
}

export async function updateRecord(id, changes) {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  const current = await new Promise((resolve, reject) => {
    const req = store.get(id);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  if (!current) throw new Error(`Record ${id} not found`);
  const updated = { ...current, ...changes, updatedAt: Date.now() };
  return new Promise((resolve, reject) => {
    const putReq = store.put(updated);
    putReq.onsuccess = () => resolve(updated);
    putReq.onerror = () => reject(putReq.error);
  });
}

export async function deleteRecord(id) {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  return new Promise((resolve, reject) => {
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

function queryByIndex(indexName, keyRange, direction) {
  const records = [];
  return openDb().then((db) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index(indexName);
    const request = index.openCursor(keyRange, direction);
    return new Promise((resolve, reject) => {
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          records.push(cursor.value);
          cursor.continue();
        } else {
          resolve(records);
        }
      };
      request.onerror = () => reject(request.error);
    });
  });
}