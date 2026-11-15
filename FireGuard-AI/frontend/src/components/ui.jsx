import { AlertTriangle, Flame, Loader2, ShieldCheck } from "lucide-react";
import { CLASS_META } from "../utils/format";

export function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`card p-5 ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatCard({ label, value, hint, icon: Icon, tone = "ai" }) {
  const tones = {
    ai: "bg-[var(--ai)]/12 text-[var(--ai)]",
    fire: "bg-[var(--fire)]/12 text-[var(--fire)]",
    safe: "bg-[var(--safe)]/12 text-[var(--safe)]",
    violet: "bg-[var(--ai-2)]/12 text-[var(--ai-2)]",
  };
  return (
    <div className="card fade-up p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-ink-3">{label}</p>
        {Icon && (
          <span className={`grid size-9 place-items-center rounded-xl ${tones[tone]}`}>
            <Icon className="size-4.5" />
          </span>
        )}
      </div>
      <p className="mt-3 text-3xl font-semibold tabular-nums text-ink">{value}</p>
      {hint && <p className="mt-1 text-xs text-ink-3">{hint}</p>}
    </div>
  );
}

/** Class label always pairs colour with an icon + text, so it never relies on colour alone. */
export function ClassBadge({ cls, size = "sm" }) {
  const meta = CLASS_META[cls];
  const Icon = cls === "fire" ? Flame : ShieldCheck;
  const sizes = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-xs";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${sizes}`}
      style={{ color: meta.color, background: `color-mix(in srgb, ${meta.color} 14%, transparent)` }}
    >
      <Icon className={size === "lg" ? "size-4" : "size-3.5"} />
      {meta.short}
    </span>
  );
}

export function Button({ variant = "primary", className = "", children, ...props }) {
  const variants = {
    primary: "bg-[var(--ai)] text-white hover:brightness-110 shadow-sm",
    secondary: "border border-default text-ink hover:bg-black/5 dark:hover:bg-white/5",
    danger: "bg-[var(--fire)] text-white hover:brightness-110",
    ghost: "text-ink-2 hover:bg-black/5 dark:hover:bg-white/5",
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Spinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 text-sm text-ink-3">
      <Loader2 className="size-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="card flex flex-col items-center gap-3 p-8 text-center">
      <AlertTriangle className="size-6 text-fire-ui" />
      <p className="text-sm text-ink-2">{message}</p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children }) {
  return (
    <div className="flex flex-col items-center gap-2 py-10 text-center">
      {Icon && <Icon className="size-8 text-ink-3" />}
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && <div className="text-xs text-ink-3">{children}</div>}
    </div>
  );
}

export function PageHeader({ title, description, actions }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 text-sm text-ink-3">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

/** Shared Recharts tooltip, styled to the card surface. */
export function ChartTooltip({ active, payload, label, formatter = (v) => v, labelFormatter = (l) => l }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-medium text-ink">{labelFormatter(label)}</p>
      {payload.map((p) => (
        <p key={p.dataKey ?? p.name} className="flex items-center gap-2 text-ink-2">
          <span className="size-2 rounded-full" style={{ background: p.color || p.payload?.fill }} />
          {p.name}: <span className="font-semibold tabular-nums text-ink">{formatter(p.value, p.name)}</span>
        </p>
      ))}
    </div>
  );
}
