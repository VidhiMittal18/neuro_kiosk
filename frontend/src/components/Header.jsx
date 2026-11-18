import { Cloud, Cpu, WifiOff } from "lucide-react";

export default function Header({ status, simulateOffline, onSimulateOfflineChange }) {
  const cloudConnected = Boolean(status?.cloud_reachable);
  const stateLabel = simulateOffline
    ? "Offline · simulated"
    : status === null
      ? "Edge service unavailable"
      : cloudConnected
        ? "Cloud connected"
        : "Cloud unreachable";
  const StateIcon = simulateOffline ? Cpu : cloudConnected ? Cloud : WifiOff;

  return (
    <header className="topbar" id="top">
      <a className="brand" href="#top" aria-label="Neuro Kiosk with NIT Agartala branding">
        <span className="brand-logos">
          <img className="brand-logo brand-logo-neuro" src="/neuro-kiosk-logo.png" alt="Neuro Kiosk logo" />
          <img className="brand-logo brand-logo-institute" src="/nit-agartala-logo.png" alt="NIT Agartala emblem" />
        </span>
        <span className="brand-wordmark">NEURO<span>KIOSK</span></span>
      </a>
      <div className="topbar-right">
        <div className={`service-state ${simulateOffline ? "state-simulated" : cloudConnected ? "state-online" : "state-offline"}`} aria-live="polite">
          <StateIcon size={15} />
          <span>{stateLabel}</span>
        </div>
        <label className="offline-switch">
          <span className="switch-copy"><span className="switch-label">Simulate offline</span><span className="switch-description">Force local SNN</span></span>
          <input
            type="checkbox"
            role="switch"
            aria-label="Simulate offline inference"
            checked={simulateOffline}
            onChange={(event) => onSimulateOfflineChange(event.target.checked)}
          />
          <span className="switch-track" aria-hidden="true"><span /></span>
        </label>
      </div>
    </header>
  );
}