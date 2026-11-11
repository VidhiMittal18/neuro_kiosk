const API_BASE = import.meta.env.DEV ? "/api" : "";

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch {
    throw new Error("Kiosk service unreachable");
  }

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP status as the actionable fallback.
    }
    throw new Error(message);
  }
  return response;
}

export async function getStatus() {
  return (await request("/status")).json();
}

export async function sendImageForDiagnosis(file, simulateOffline) {
  const form = new FormData();
  form.append("file", file);
  form.append("simulate_offline", String(simulateOffline));
  return (await request("/diagnose", { method: "POST", body: form })).json();
}

export async function diagnose(file, simulateOffline) {
  return sendImageForDiagnosis(file, simulateOffline);
}

export async function getBenchmarks() {
  return (await request("/benchmarks")).json();
}

export async function getSampleFile(filename) {
  const response = await request(`/samples/${encodeURIComponent(filename)}`);
  const blob = await response.blob();
  return new File([blob], filename, { type: "image/png" });
}