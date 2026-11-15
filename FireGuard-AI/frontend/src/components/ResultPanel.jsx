import { useState } from "react";
import { AlertTriangle, Clock, Cpu, Download, Fingerprint, ShieldCheck } from "lucide-react";
import ProbabilityBars from "./ProbabilityBars";
import { Button } from "./ui";
import { api, assetUrl } from "../services/api";
import { dateTime, prob } from "../utils/format";

/** Full analysis result: status banner, confidence, probabilities, image + Grad-CAM views. */
export default function ResultPanel({ result }) {
  const isFire = result.predicted_class === "fire";
  const accent = isFire ? "var(--fire)" : "var(--safe)";
  const views = [
    { key: "original", label: "Original", url: assetUrl(result.image_url) },
    result.overlay_url && { key: "overlay", label: "Grad-CAM overlay", url: assetUrl(result.overlay_url) },
    result.heatmap_url && { key: "heatmap", label: "Heatmap", url: assetUrl(result.heatmap_url) },
  ].filter(Boolean);
  const [view, setView] = useState(views[0].key);
  const current = views.find((v) => v.key === view) ?? views[0];

  return (
    <div className="fade-up space-y-5">
      <div
        className="relative overflow-hidden rounded-2xl border p-5"
        style={{
          borderColor: `color-mix(in srgb, ${accent} 45%, transparent)`,
          background: `linear-gradient(135deg, color-mix(in srgb, ${accent} 18%, var(--card)), var(--card) 70%)`,
        }}
        role="status"
      >
        <div className="flex flex-wrap items-center gap-4">
          <span
            className={`grid size-14 place-items-center rounded-2xl text-white shadow-lg ${isFire ? "animate-pulse" : ""}`}
            style={{ background: accent }}
          >
            {isFire ? <AlertTriangle className="size-7" /> : <ShieldCheck className="size-7" />}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: accent }}>
              {isFire ? "Warning" : "All clear"}
            </p>
            <p className="text-2xl font-bold tracking-tight text-ink">{result.label}</p>
            <p className="text-sm text-ink-2">
              {isFire
                ? "The model found visual evidence of fire in this image. Verify and respond immediately."
                : "The model found no visual evidence of fire in this image."}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs uppercase tracking-wide text-ink-3">Confidence</p>
            <p className="text-4xl font-bold tabular-nums" style={{ color: accent }}>
              {prob(result.confidence, 1)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div className="mb-2 flex flex-wrap gap-1.5">
            {views.map((v) => (
              <button
                key={v.key}
                onClick={() => setView(v.key)}
                className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
                  view === v.key ? "bg-[var(--ai)] text-white" : "text-ink-2 hover:bg-black/5 dark:hover:bg-white/5"
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>
          <div className="overflow-hidden rounded-xl border border-default bg-black/5 dark:bg-black/30">
            <img src={current.url} alt={current.label} className="mx-auto max-h-[420px] w-full object-contain" />
          </div>
          {view !== "original" && (
            <p className="mt-2 text-xs text-ink-3">
              Grad-CAM: warmer regions influenced the “{result.label.toLowerCase()}” decision the most.
            </p>
          )}
        </div>

        <div className="space-y-5 lg:col-span-2">
          <div className="card p-4">
            <ProbabilityBars
              fire={result.fire_probability}
              noFire={result.no_fire_probability}
              threshold={result.threshold}
            />
          </div>
          <dl className="card divide-y divide-[var(--card-border)] text-sm">
            {[
              [Fingerprint, "Prediction ID", <span className="break-all font-mono text-xs">{result.id}</span>],
              [Clock, "Analysed", dateTime(result.created_at)],
              [Cpu, "Inference", `${result.inference_ms.toFixed(1)} ms · model v${result.model_version}`],
            ].map(([Icon, label, value]) => (
              <div key={label} className="flex items-start justify-between gap-3 px-4 py-2.5">
                <dt className="flex shrink-0 items-center gap-2 text-ink-3">
                  <Icon className="size-4" />
                  {label}
                </dt>
                <dd className="text-right text-ink">{value}</dd>
              </div>
            ))}
          </dl>
          <a href={api.reportUrl(result.id)} download>
            <Button variant="secondary" className="w-full">
              <Download className="size-4" /> Download PDF report
            </Button>
          </a>
        </div>
      </div>
    </div>
  );
}
