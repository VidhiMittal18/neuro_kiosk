import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/700.css";
import App from "./App.jsx";
import "./styles.css";
import { registerBackgroundSync } from "./sync.js";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

async function bootstrapServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  try {
    const serviceWorkerUrl = import.meta.env.DEV ? "/sw.js?apiBase=%2Fapi" : "/sw.js";
    await navigator.serviceWorker.register(serviceWorkerUrl, { scope: "/" });
    await registerBackgroundSync();
  } catch {
    // Service worker unavailable; sync relies on the online event only.
  }
}

bootstrapServiceWorker();