import { useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Flame, Gauge, ImageIcon, ScanSearch, ShieldCheck, Timer } from "lucide-react";
import { Button, Card, ChartTooltip, ClassBadge, EmptyState, ErrorState, PageHeader, Spinner, StatCard } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { api, assetUrl } from "../services/api";
import { dateTime, num, pct, prob, shortDay } from "../utils/format";

const RANGES = [7, 14, 30];

function LegendLabel(value) {
  return <span className="text-xs text-ink-2">{value}</span>;
}

export default function Overview() {
  const [days, setDays] = useState(14);
  const { data, loading, error, reload } = useAsync(() => api.analytics(days), [days]);

  if (loading && !data) return <Spinner label="Loading analytics…" />;
  if (error && !data) return <ErrorState message={error} onRetry={reload} />;

  const empty = data.total_predictions === 0;
  const distribution = [
    { name: "Fire", value: data.fire_detections, fill: "var(--fire)" },
    { name: "No fire", value: data.non_fire_detections, fill: "var(--safe)" },
  ];

  return (
    <>
      <PageHeader
        title="Overview"
        description="Live statistics computed from every image analysed by the model."
        actions={
          <Link to="/detect">
            <Button>
              <ScanSearch className="size-4" /> Analyse an image
            </Button>
          </Link>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Images analysed" value={num(data.total_predictions)} icon={ImageIcon} tone="ai"
                  hint={`${num(data.predictions_last_24h)} in the last 24 h`} />
        <StatCard label="Fire cases detected" value={num(data.fire_detections)} icon={Flame} tone="fire"
                  hint={`${num(data.fire_last_24h)} in the last 24 h`} />
        <StatCard label="Safe images" value={num(data.non_fire_detections)} icon={ShieldCheck} tone="safe"
                  hint="Classified as no fire" />
        <StatCard label="Fire detection rate" value={`${data.fire_detection_rate.toFixed(1)}%`} icon={Gauge} tone="violet"
                  hint={data.average_confidence != null ? `Avg. confidence ${pct(data.average_confidence)}` : "No data yet"} />
      </div>

      {empty ? (
        <Card className="mt-6">
          <EmptyState icon={ScanSearch} title="No images analysed yet">
            Upload your first image on the <Link className="text-[var(--ai)] underline" to="/detect">Image Detection</Link> page — the dashboard fills in from real predictions.
          </EmptyState>
        </Card>
      ) : (
        <>
          <div className="mt-6 grid gap-5 xl:grid-cols-3">
            <Card
              className="xl:col-span-2"
              title="Prediction trends"
              subtitle="Daily analyses by predicted class (UTC days)"
              action={
                <div className="flex rounded-lg border border-default p-0.5 text-xs">
                  {RANGES.map((r) => (
                    <button key={r} onClick={() => setDays(r)}
                            className={`rounded-md px-2.5 py-1 font-medium ${days === r ? "bg-[var(--ai)] text-white" : "text-ink-2"}`}>
                      {r}d
                    </button>
                  ))}
                </div>
              }
            >
              <div className="h-72">
                <ResponsiveContainer>
                  {/* Stacked daily bars: bar height = analyses that day, segments = class split. */}
                  <BarChart data={data.daily_trend} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--grid)" />
                    <XAxis dataKey="date" tickFormatter={shortDay} tickLine={false} axisLine={false} minTickGap={24} />
                    <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip labelFormatter={shortDay} />} />
                    <Legend iconType="circle" formatter={LegendLabel} />
                    <Bar name="No fire" dataKey="non_fire" stackId="day" fill="var(--safe)" maxBarSize={28}
                         stroke="var(--card)" strokeWidth={1} />
                    <Bar name="Fire" dataKey="fire" stackId="day" fill="var(--fire)" maxBarSize={28} radius={[4, 4, 0, 0]}
                         stroke="var(--card)" strokeWidth={1} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title="Fire vs non-fire" subtitle="Share of all predictions">
              <div className="relative h-56">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={distribution} dataKey="value" nameKey="name" innerRadius="64%" outerRadius="92%"
                         paddingAngle={2} stroke="var(--card)" strokeWidth={2}
                         // The Pie entry animation can stall on its first frame (seen in testing); draw it directly.
                         isAnimationActive={false}>
                      {distribution.map((d) => <Cell key={d.name} fill={d.fill} />)}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
                  <div>
                    <p className="text-2xl font-semibold tabular-nums text-ink">{num(data.total_predictions)}</p>
                    <p className="text-xs text-ink-3">images</p>
                  </div>
                </div>
              </div>
              <div className="mt-3 space-y-2">
                {distribution.map((d) => (
                  <div key={d.name} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2 text-ink-2">
                      {d.name === "Fire" ? <Flame className="size-4" style={{ color: d.fill }} /> : <ShieldCheck className="size-4" style={{ color: d.fill }} />}
                      {d.name}
                    </span>
                    <span className="tabular-nums text-ink">
                      {num(d.value)} <span className="text-ink-3">({pct(d.value / data.total_predictions)})</span>
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-3">
            <Card title="Confidence distribution" subtitle="Predictions per confidence band (%), by predicted class">
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={data.confidence_distribution} margin={{ top: 8, right: 4, left: -20, bottom: 0 }} barGap={2}>
                    <CartesianGrid vertical={false} stroke="var(--grid)" />
                    <XAxis dataKey="range" tickLine={false} axisLine={false} interval={0}
                           tickFormatter={(r) => r.replace("%", "")} />
                    <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend iconType="circle" formatter={LegendLabel} />
                    <Bar name="Fire" dataKey="fire" fill="var(--fire)" radius={[4, 4, 0, 0]} maxBarSize={18} />
                    <Bar name="No fire" dataKey="non_fire" fill="var(--safe)" radius={[4, 4, 0, 0]} maxBarSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card className="xl:col-span-2" title="Recent analysis activity"
                  action={<Link to="/history" className="text-xs font-medium text-[var(--ai)]">View all</Link>}>
              <ul className="divide-y divide-[var(--card-border)]">
                {data.recent_predictions.map((p) => (
                  <li key={p.id} className="flex items-center gap-3 py-2.5">
                    <img src={assetUrl(p.thumbnail_url)} alt="" className="size-11 rounded-lg object-cover" loading="lazy" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-ink">{p.filename}</p>
                      <p className="text-xs text-ink-3">{dateTime(p.created_at)}</p>
                    </div>
                    <ClassBadge cls={p.predicted_class} />
                    <span className="hidden w-16 text-right text-sm tabular-nums text-ink sm:block">{prob(p.confidence, 1)}</span>
                  </li>
                ))}
              </ul>
              {data.average_inference_ms != null && (
                <p className="mt-3 flex items-center gap-1.5 text-xs text-ink-3">
                  <Timer className="size-3.5" /> Average model inference time {data.average_inference_ms.toFixed(1)} ms
                </p>
              )}
            </Card>
          </div>
        </>
      )}
    </>
  );
}
