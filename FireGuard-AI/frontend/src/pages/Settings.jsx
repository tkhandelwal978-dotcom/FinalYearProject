import { useOutletContext } from "react-router-dom";
import { CheckCircle2, Info, Moon, Server, Sun, XCircle } from "lucide-react";
import { Card, PageHeader } from "../components/ui";
import { API_URL } from "../services/api";

function StatusRow({ label, ok, value }) {
  return (
    <div className="flex items-center justify-between gap-3 py-3">
      <span className="text-sm text-ink-2">{label}</span>
      <span className="flex items-center gap-1.5 text-sm font-medium text-ink">
        {ok === true && <CheckCircle2 className="size-4 text-safe-ui" />}
        {ok === false && <XCircle className="size-4 text-fire-ui" />}
        {value}
      </span>
    </div>
  );
}

export default function SettingsPage() {
  const { theme, health } = useOutletContext();
  const h = health.data;

  return (
    <>
      <PageHeader title="Settings" description="Appearance, system status and application information." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Appearance">
          <div className="grid grid-cols-2 gap-3">
            {[
              { value: "light", label: "Light", icon: Sun },
              { value: "dark", label: "Dark", icon: Moon },
            ].map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                onClick={() => theme.setTheme(value)}
                className={`flex items-center gap-3 rounded-xl border p-4 text-left transition ${
                  theme.theme === value ? "border-[var(--ai)] bg-[var(--ai)]/8" : "border-default hover:bg-black/5 dark:hover:bg-white/5"
                }`}
                aria-pressed={theme.theme === value}
              >
                <Icon className="size-5 text-ink-2" />
                <span className="text-sm font-medium text-ink">{label}</span>
              </button>
            ))}
          </div>
        </Card>

        <Card title="System status" subtitle="Live from /api/health (refreshes every 30 s)">
          <div className="divide-y divide-[var(--card-border)]">
            <StatusRow label="Backend" ok={health.status !== "offline" && health.status !== "checking"}
                       value={health.status === "offline" ? "Unreachable" : health.status === "checking" ? "Checking…" : "Online"} />
            <StatusRow label="Database" ok={h ? h.database === "ok" : undefined} value={h?.database ?? "—"} />
            <StatusRow label="Model" ok={h ? h.model_loaded : undefined} value={h ? (h.model_loaded ? "Loaded" : "Not loaded") : "—"} />
            <StatusRow label="Model version" value={h?.model_version ? `v${h.model_version}` : "—"} />
            <StatusRow label="Inference device" value={h?.device ?? "—"} />
          </div>
          {(health.error || h?.model_error) && (
            <p className="mt-3 rounded-xl bg-[var(--fire)]/8 p-3 text-xs text-ink-2">{health.error || h.model_error}</p>
          )}
        </Card>

        <Card title="Application" className="lg:col-span-2">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="flex items-start gap-3">
              <Info className="mt-0.5 size-4 text-[var(--ai)]" />
              <div className="text-sm">
                <p className="font-medium text-ink">{h?.app_name ?? "FireGuard AI"}</p>
                <p className="text-ink-3">Version {h?.app_version ?? "—"}</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Server className="mt-0.5 size-4 text-[var(--ai)]" />
              <div className="text-sm">
                <p className="font-medium text-ink">API endpoint</p>
                <a className="break-all text-ink-3 hover:text-[var(--ai)]" href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
                  {API_URL} (Swagger docs)
                </a>
              </div>
            </div>
            <p className="text-sm text-ink-3">
              FireGuard AI is a decision-support tool. Predictions can be wrong; always confirm alerts and never use it in
              place of certified fire alarm systems.
            </p>
          </div>
        </Card>
      </div>
    </>
  );
}
