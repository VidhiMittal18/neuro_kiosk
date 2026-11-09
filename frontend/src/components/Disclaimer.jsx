import { Info } from "lucide-react";

export default function Disclaimer() {
  return (
    <div className="disclaimer" role="note">
      <Info size={15} aria-hidden="true" />
      <span><strong>Research demo.</strong> Not for clinical use. Not a medical device.</span>
    </div>
  );
}