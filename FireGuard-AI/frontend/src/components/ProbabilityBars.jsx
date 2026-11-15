import { Flame, ShieldCheck } from "lucide-react";
import { pct, prob } from "../utils/format";

/** Fire vs no-fire probabilities as two labelled meters, with the decision threshold marked. */
export default function ProbabilityBars({ fire, noFire, threshold }) {
  const rows = [
    { key: "fire", label: "Probability of fire", value: fire, color: "var(--fire)", icon: Flame },
    { key: "non_fire", label: "Probability of no fire", value: noFire, color: "var(--safe)", icon: ShieldCheck },
  ];
  return (
    <div className="space-y-4">
      {rows.map(({ key, label, value, color, icon: Icon }) => (
        <div key={key}>
          <div className="mb-1.5 flex items-center justify-between text-sm">
            <span className="flex items-center gap-1.5 text-ink-2">
              <Icon className="size-4" style={{ color }} />
              {label}
            </span>
            <span className="font-semibold tabular-nums text-ink">{prob(value)}</span>
          </div>
          <div className="relative h-2.5 overflow-hidden rounded-full bg-black/8 dark:bg-white/8">
            <div
              className="h-full rounded-full transition-[width] duration-700 ease-out"
              style={{ width: `${Math.max(value * 100, 0.5)}%`, background: color }}
            />
            {key === "fire" && threshold != null && (
              <div
                className="absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-[var(--ink)]"
                style={{ left: `${threshold * 100}%` }}
                title={`Decision threshold ${pct(threshold, 0)}`}
              />
            )}
          </div>
        </div>
      ))}
      {threshold != null && (
        <p className="text-xs text-ink-3">
          The marker shows the decision threshold ({pct(threshold, 0)} fire probability), tuned on validation data to keep
          fire recall high.
        </p>
      )}
    </div>
  );
}
