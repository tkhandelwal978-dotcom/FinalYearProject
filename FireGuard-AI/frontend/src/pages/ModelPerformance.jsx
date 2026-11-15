import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Award, BrainCircuit, Crosshair, Database, Flame, Layers, Target, TrendingUp } from "lucide-react";
import { Card, ChartTooltip, ErrorState, PageHeader, Spinner, StatCard } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { api, assetUrl } from "../services/api";
import { num, pct } from "../utils/format";

const legendLabel = (value) => <span className="text-xs text-ink-2">{value}</span>;

function HistoryChart({ history, metric, title, subtitle, fineTuneStart }) {
  const isLoss = metric === "loss";
  return (
    <Card title={title} subtitle={subtitle}>
      <div className="h-64">
        <ResponsiveContainer>
          <LineChart data={history} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
            <YAxis tickLine={false} axisLine={false} domain={isLoss ? [0, "auto"] : ["auto", 1]}
                   tickFormatter={(v) => (isLoss ? v.toFixed(2) : `${Math.round(v * 100)}%`)} />
            <Tooltip content={<ChartTooltip labelFormatter={(e) => `Epoch ${e}`}
                                            formatter={(v) => (isLoss ? v.toFixed(4) : pct(v, 2))} />} />
            <Legend iconType="circle" formatter={legendLabel} />
            {fineTuneStart && (
              <ReferenceLine x={fineTuneStart} stroke="var(--ink-3)" strokeDasharray="4 4"
                             label={{ value: "fine-tuning", position: "insideTopLeft", fill: "var(--ink-3)", fontSize: 11 }} />
            )}
            <Line name="Training" dataKey={`train_${metric}`} stroke="var(--ai)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
            <Line name="Validation" dataKey={`val_${metric}`} stroke="var(--ai-2)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function CurveChart({ title, subtitle, data, xKey, yKey, xLabel, yLabel, color, diagonal }) {
  return (
    <Card title={title} subtitle={subtitle}>
      <div className="h-64">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 12 }}>
            <CartesianGrid stroke="var(--grid)" />
            <XAxis type="number" dataKey={xKey} domain={[0, 1]} tickCount={6} tickLine={false} axisLine={false}
                   label={{ value: xLabel, position: "insideBottom", offset: -6, fill: "var(--ink-3)", fontSize: 11 }} />
            <YAxis type="number" domain={[0, 1]} tickCount={6} tickLine={false} axisLine={false}
                   label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 18, fill: "var(--ink-3)", fontSize: 11 }} />
            <Tooltip content={<ChartTooltip labelFormatter={(x) => `${xLabel}: ${Number(x).toFixed(3)}`} formatter={(v) => Number(v).toFixed(3)} />} />
            {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--ink-3)" strokeDasharray="4 4" />}
            <Line name={yLabel} dataKey={yKey} stroke={color} strokeWidth={2} dot={false} type="stepAfter" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function ConfusionMatrix({ cm }) {
  const [[tn, fp], [fn, tp]] = cm.matrix;
  const max = Math.max(tn, fp, fn, tp);
  const cells = [
    { label: "True negative", value: tn, good: true, row: "No fire", col: "No fire" },
    { label: "False positive", value: fp, good: false, row: "No fire", col: "Fire" },
    { label: "False negative (missed fire)", value: fn, good: false, row: "Fire", col: "No fire" },
    { label: "True positive", value: tp, good: true, row: "Fire", col: "Fire" },
  ];
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
        <div />
        <p className="text-center text-xs font-medium text-ink-3">Predicted: No fire</p>
        <p className="text-center text-xs font-medium text-ink-3">Predicted: Fire</p>
        {[0, 1].map((r) => (
          <div key={r} className="contents">
            <p className="flex items-center pr-1 text-xs font-medium text-ink-3 [writing-mode:vertical-rl] rotate-180">
              Actual: {r === 0 ? "No fire" : "Fire"}
            </p>
            {cells.slice(r * 2, r * 2 + 2).map((c) => (
              <div
                key={c.label}
                className="rounded-xl p-4 text-center"
                style={{
                  background: `color-mix(in srgb, ${c.good ? "var(--ai)" : "var(--fire)"} ${10 + (c.value / max) * 45}%, var(--card))`,
                }}
                title={c.label}
              >
                <p className="text-2xl font-semibold tabular-nums text-ink">{num(c.value)}</p>
                <p className="text-[11px] text-ink-2">{c.label}</p>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModelPerformance() {
  const { data, loading, error, reload } = useAsync(() => api.modelInfo(), []);
  if (loading && !data) return <Spinner label="Loading model artifacts…" />;
  if (error && !data) return <ErrorState message={error} onRetry={reload} />;

  const m = data.test_metrics;
  const history = data.training_history || [];
  const fineTuneStart = history.find((h) => h.phase === "fine_tune")?.epoch;
  const ds = data.dataset;
  const roc = data.curves ? data.curves.roc.fpr.map((x, i) => ({ fpr: x, tpr: data.curves.roc.tpr[i] })) : [];
  const pr = data.curves
    ? data.curves.pr.recall.map((x, i) => ({ recall: x, precision: data.curves.pr.precision[i] })).sort((a, b) => a.recall - b.recall)
    : [];

  return (
    <>
      <PageHeader
        title="Model Performance"
        description="All figures are read from the training and evaluation artifacts — measured on a held-out test set the model never saw."
      />

      {!m ? (
        <ErrorState message="No evaluation results found. Run `python scripts/evaluate.py` to generate test metrics." onRetry={reload} />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            <StatCard label="Test accuracy" value={pct(m.accuracy, 2)} icon={Award} tone="ai" hint={`${num(m.test_samples)} test images`} />
            <StatCard label="Precision" value={pct(m.precision, 2)} icon={Crosshair} tone="violet" hint={`${m.false_positives_false_alarms} false alarms`} />
            <StatCard label="Recall (fire)" value={pct(m.recall, 2)} icon={Flame} tone="fire" hint={`${m.false_negatives_missed_fires} missed fires`} />
            <StatCard label="F1 score" value={pct(m.f1_score, 2)} icon={Target} tone="ai" hint={`Threshold ${m.decision_threshold.toFixed(2)}`} />
            <StatCard label="ROC-AUC" value={m.roc_auc.toFixed(4)} icon={TrendingUp} tone="safe" hint={`Avg. precision ${m.average_precision.toFixed(4)}`} />
          </div>

          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <Card title="Confusion matrix" subtitle={`Test set · decision threshold ${m.decision_threshold.toFixed(2)}`}>
              <ConfusionMatrix cm={m.confusion_matrix} />
            </Card>
            <Card title="Architecture" subtitle={data.framework}>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                {[
                  ["Model", `${data.model_name} v${data.model_version}`],
                  ["Input size", data.image_size?.join(" × ")],
                  ["Parameters", num(data.training?.total_parameters)],
                  ["Fine-tuned params", num(data.training?.trainable_parameters_fine_tune)],
                  ["Epochs (best)", `${data.training?.epochs_run} (${data.training?.best_epoch})`],
                  ["Training time", `${data.training?.training_duration_minutes} min`],
                  ["Batch size", data.hyperparameters?.batch_size],
                  ["Dropout", data.hyperparameters?.dropout],
                  ["Learning rates", `${data.hyperparameters?.head_lr} → ${data.hyperparameters?.fine_tune_lr}`],
                  ["Unfrozen blocks", data.hyperparameters?.fine_tune_blocks],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-xs text-ink-3">{k}</dt>
                    <dd className="font-medium text-ink">{v ?? "—"}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-4 flex items-start gap-2 rounded-xl bg-[var(--ai)]/8 p-3 text-xs text-ink-2">
                <Layers className="mt-0.5 size-4 shrink-0 text-[var(--ai)]" />
                {data.architecture}
              </p>
            </Card>
          </div>
        </>
      )}

      {history.length > 0 && (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <HistoryChart history={history} metric="accuracy" title="Training accuracy" subtitle="Training vs validation, per epoch" fineTuneStart={fineTuneStart} />
          <HistoryChart history={history} metric="loss" title="Training loss" subtitle="Binary cross-entropy, per epoch" fineTuneStart={fineTuneStart} />
        </div>
      )}

      {data.curves && m && (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <CurveChart title="ROC curve" subtitle={`AUC ${m.roc_auc.toFixed(4)} · test set`} data={roc} xKey="fpr" yKey="tpr"
                      xLabel="False positive rate" yLabel="True positive rate" color="var(--fire)" diagonal />
          <CurveChart title="Precision-recall curve" subtitle={`Average precision ${m.average_precision.toFixed(4)}`} data={pr}
                      xKey="recall" yKey="precision" xLabel="Recall" yLabel="Precision" color="var(--ai-2)" />
        </div>
      )}

      {ds && (
        <Card className="mt-5" title="Training dataset" subtitle={`Seed ${ds.seed} · de-duplicated with perceptual hashing before splitting (no leakage)`}>
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-default text-left text-xs uppercase tracking-wide text-ink-3">
                    <th className="py-2 font-medium">Split</th>
                    <th className="py-2 text-right font-medium">Fire</th>
                    <th className="py-2 text-right font-medium">No fire</th>
                    <th className="py-2 text-right font-medium">Total</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--card-border)]">
                  {Object.entries(ds.split_counts).map(([split, c]) => (
                    <tr key={split}>
                      <td className="py-2 capitalize text-ink">{split}</td>
                      <td className="py-2 text-right tabular-nums text-ink">{num(c.fire)}</td>
                      <td className="py-2 text-right tabular-nums text-ink">{num(c.non_fire)}</td>
                      <td className="py-2 text-right tabular-nums font-medium text-ink">{num(c.fire + c.non_fire)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <ul className="mt-3 space-y-1 text-xs text-ink-3">
                <li>Raw images found: {num(ds.raw_images_found)}</li>
                <li>Duplicates / near-duplicates removed: {num(ds.duplicates_removed)}</li>
                <li>Corrupted or too-small removed: {num(ds.corrupted_or_too_small_removed)}</li>
                <li>Conflicting-label images removed: {num(ds.conflicting_label_images_removed)}</li>
              </ul>
            </div>
            <ul className="space-y-3">
              {ds.sources.map((s) => (
                <li key={s.key} className="flex items-start gap-3 rounded-xl border border-default p-3">
                  <Database className="mt-0.5 size-4 shrink-0 text-[var(--ai)]" />
                  <div className="min-w-0 text-sm">
                    <a href={s.url} target="_blank" rel="noreferrer" className="font-medium text-ink hover:text-[var(--ai)]">{s.title}</a>
                    <p className="text-xs text-ink-3">{s.license} · {num(s.images_used)} images used</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      )}

      {Object.keys(data.plots || {}).length > 0 && (
        <Card className="mt-5" title="Evaluation plots" subtitle="Generated by scripts/train.py and scripts/evaluate.py (artifacts/plots)">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {Object.entries(data.plots).map(([key, url]) => (
              <a key={key} href={assetUrl(url)} target="_blank" rel="noreferrer" className="group overflow-hidden rounded-xl border border-default bg-white">
                <img src={assetUrl(url)} alt={key.replaceAll("_", " ")} loading="lazy" className="aspect-[4/3] w-full object-contain transition group-hover:scale-[1.02]" />
                <p className="border-t border-default bg-[var(--card)] px-3 py-2 text-xs capitalize text-ink-2">{key.replaceAll("_", " ")}</p>
              </a>
            ))}
          </div>
        </Card>
      )}

      {!data.model_loaded && (
        <p className="mt-5 flex items-center gap-2 text-sm text-fire-ui">
          <BrainCircuit className="size-4" /> The backend could not load the model checkpoint. Check the Settings page for details.
        </p>
      )}
    </>
  );
}
