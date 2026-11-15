import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Activity, BarChart3, BrainCircuit, Flame, History, Menu, Moon, ScanSearch, Settings, Sun, X } from "lucide-react";
import { useHealth } from "../hooks/useHealth";
import { useTheme } from "../hooks/useTheme";

const NAV = [
  { to: "/", label: "Overview", icon: BarChart3, end: true },
  { to: "/detect", label: "Image Detection", icon: ScanSearch },
  { to: "/history", label: "Prediction History", icon: History },
  { to: "/model", label: "Model Performance", icon: BrainCircuit },
  { to: "/settings", label: "Settings", icon: Settings },
];

function HealthPill({ health }) {
  const states = {
    ok: { text: "Model online", color: "var(--safe)" },
    degraded: { text: "Model unavailable", color: "#d97706" },
    offline: { text: "Backend offline", color: "var(--fire)" },
    checking: { text: "Connecting…", color: "var(--ink-3)" },
  };
  const s = states[health.status] ?? states.checking;
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-default px-3 py-1 text-xs font-medium text-ink-2">
      <span className="relative flex size-2">
        {health.status === "ok" && (
          <span className="absolute inline-flex size-full animate-ping rounded-full opacity-60" style={{ background: s.color }} />
        )}
        <span className="relative inline-flex size-2 rounded-full" style={{ background: s.color }} />
      </span>
      {s.text}
    </span>
  );
}

export default function AppLayout() {
  const theme = useTheme();
  const health = useHealth();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const current = NAV.find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to)));

  return (
    <div className="min-h-screen bg-surface">
      {open && <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-default bg-[var(--card)] transition-transform lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 px-5 py-5">
          <span className="grid size-10 place-items-center rounded-xl bg-gradient-to-br from-[#f05252] to-[#f59e0b] text-white shadow-lg shadow-red-500/20">
            <Flame className="size-5" />
          </span>
          <div>
            <p className="text-base font-bold tracking-tight text-ink">FireGuard AI</p>
            <p className="text-[11px] text-ink-3">Intelligent fire detection</p>
          </div>
          <button className="ml-auto lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
            <X className="size-5 text-ink-3" />
          </button>
        </div>
        <nav className="flex-1 space-y-1 px-3">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                  isActive
                    ? "bg-[var(--ai)]/12 text-[var(--ai)]"
                    : "text-ink-2 hover:bg-black/5 hover:text-ink dark:hover:bg-white/5"
                }`
              }
            >
              <Icon className="size-4.5" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="m-3 rounded-xl border border-default p-3 text-xs text-ink-3">
          <p className="flex items-center gap-1.5 font-medium text-ink-2">
            <Activity className="size-3.5" /> EfficientNet-B0
          </p>
          <p className="mt-1">
            {health.data?.model_version ? `Model v${health.data.model_version} · ${health.data.device}` : "Model status unknown"}
          </p>
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-default bg-[var(--surface)]/85 px-4 py-3 backdrop-blur sm:px-8">
          <button className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu className="size-5 text-ink-2" />
          </button>
          <p className="text-sm font-medium text-ink-2">{current?.label}</p>
          <div className="ml-auto flex items-center gap-2">
            <HealthPill health={health} />
            <button
              onClick={theme.toggle}
              className="grid size-9 place-items-center rounded-xl border border-default text-ink-2 transition hover:text-ink"
              aria-label="Toggle theme"
            >
              {theme.theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 sm:px-8 sm:py-8">
          <Outlet context={{ theme, health }} />
        </main>
      </div>
    </div>
  );
}
