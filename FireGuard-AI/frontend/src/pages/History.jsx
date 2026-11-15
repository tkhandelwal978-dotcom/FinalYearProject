import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Download, Eye, FileDown, History as HistoryIcon, Search, Trash2 } from "lucide-react";
import Modal from "../components/Modal";
import ResultPanel from "../components/ResultPanel";
import { Button, Card, ClassBadge, EmptyState, ErrorState, PageHeader, Spinner } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { api, assetUrl, errorMessage } from "../services/api";
import { dateTime, prob } from "../utils/format";

const FILTERS = [
  { value: "", label: "All" },
  { value: "fire", label: "Fire" },
  { value: "non_fire", label: "No fire" },
];

export default function HistoryPage() {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState("newest");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [viewing, setViewing] = useState(null);
  const [deleting, setDeleting] = useState(null);
  const [actionError, setActionError] = useState(null);

  // Debounce the search box so typing doesn't fire a request per keystroke.
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const { data, loading, error, reload } = useAsync(
    () => api.history({ page, page_size: pageSize, predicted_class: filter || undefined, search: search || undefined, sort }),
    [page, pageSize, filter, search, sort],
  );

  const confirmDelete = async () => {
    try {
      await api.deletePrediction(deleting.id);
      setDeleting(null);
      if (data.items.length === 1 && page > 1) setPage(page - 1);
      else reload();
    } catch (err) {
      setActionError(errorMessage(err));
      setDeleting(null);
    }
  };

  return (
    <>
      <PageHeader
        title="Prediction History"
        description="Every analysed image, stored with its prediction and model version."
        actions={
          <a href={api.exportUrl(filter)} download>
            <Button variant="secondary">
              <FileDown className="size-4" /> Export CSV
            </Button>
          </a>
        }
      />

      <Card>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <div className="relative min-w-[220px] flex-1">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-3" />
            <input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search by filename or prediction ID"
              className="w-full rounded-xl border border-default bg-transparent py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-[var(--ink-3)] focus:border-[var(--ai)]"
            />
          </div>
          <div className="flex rounded-xl border border-default p-0.5 text-sm">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => {
                  setFilter(f.value);
                  setPage(1);
                }}
                className={`rounded-lg px-3 py-1.5 font-medium ${filter === f.value ? "bg-[var(--ai)] text-white" : "text-ink-2"}`}
              >
                {f.label}
              </button>
            ))}
          </div>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="rounded-xl border border-default bg-[var(--card)] px-3 py-2 text-sm text-ink"
            aria-label="Sort"
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="confidence">Highest confidence</option>
            <option value="fire_probability">Highest fire probability</option>
          </select>
        </div>

        {actionError && <p className="mb-3 text-sm text-fire-ui">{actionError}</p>}

        {error && !data ? (
          <ErrorState message={error} onRetry={reload} />
        ) : !data ? (
          <Spinner />
        ) : data.items.length === 0 ? (
          <EmptyState icon={HistoryIcon} title={search || filter ? "No predictions match these filters" : "No predictions yet"}>
            {search || filter ? "Try a different search or filter." : "Analysed images will appear here."}
          </EmptyState>
        ) : (
          <div className={`overflow-x-auto transition-opacity ${loading ? "opacity-60" : ""}`}>
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead>
                <tr className="border-b border-default text-xs uppercase tracking-wide text-ink-3">
                  <th className="py-2.5 pr-3 font-medium">Image</th>
                  <th className="px-3 py-2.5 font-medium">Prediction ID</th>
                  <th className="px-3 py-2.5 font-medium">Result</th>
                  <th className="px-3 py-2.5 text-right font-medium">Confidence</th>
                  <th className="px-3 py-2.5 font-medium">Timestamp</th>
                  <th className="py-2.5 pl-3 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--card-border)]">
                {data.items.map((p) => (
                  <tr key={p.id} className="hover:bg-black/[0.02] dark:hover:bg-white/[0.02]">
                    <td className="py-2.5 pr-3">
                      <div className="flex items-center gap-3">
                        <img src={assetUrl(p.thumbnail_url)} alt="" loading="lazy" className="size-12 rounded-lg object-cover" />
                        <span className="max-w-[180px] truncate text-ink" title={p.filename}>{p.filename}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-xs text-ink-2" title={p.id}>{p.id.slice(0, 8)}…</td>
                    <td className="px-3 py-2.5"><ClassBadge cls={p.predicted_class} /></td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-ink">{prob(p.confidence, 1)}</td>
                    <td className="px-3 py-2.5 text-ink-2">{dateTime(p.created_at)}</td>
                    <td className="py-2.5 pl-3">
                      <div className="flex justify-end gap-1">
                        <button onClick={() => setViewing(p)} className="rounded-lg p-2 text-ink-2 hover:bg-[var(--ai)]/10 hover:text-[var(--ai)]" title="View details" aria-label="View details">
                          <Eye className="size-4" />
                        </button>
                        <a href={api.reportUrl(p.id)} download className="rounded-lg p-2 text-ink-2 hover:bg-[var(--ai)]/10 hover:text-[var(--ai)]" title="Download report" aria-label="Download report">
                          <Download className="size-4" />
                        </a>
                        <button onClick={() => setDeleting(p)} className="rounded-lg p-2 text-ink-2 hover:bg-[var(--fire)]/10 hover:text-[var(--fire)]" title="Delete" aria-label="Delete">
                          <Trash2 className="size-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data && data.total > 0 && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-ink-2">
            <span>
              Showing {(data.page - 1) * data.page_size + 1}–{Math.min(data.page * data.page_size, data.total)} of {data.total}
            </span>
            <div className="flex items-center gap-2">
              <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}
                      className="rounded-lg border border-default bg-[var(--card)] px-2 py-1 text-sm text-ink" aria-label="Rows per page">
                {[10, 25, 50].map((n) => <option key={n} value={n}>{n} / page</option>)}
              </select>
              <Button variant="secondary" className="px-2.5" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page">
                <ChevronLeft className="size-4" />
              </Button>
              <span className="tabular-nums">Page {data.page} of {data.pages}</span>
              <Button variant="secondary" className="px-2.5" disabled={page >= data.pages} onClick={() => setPage(page + 1)} aria-label="Next page">
                <ChevronRight className="size-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>

      <Modal open={!!viewing} onClose={() => setViewing(null)} title="Prediction details" wide>
        {viewing && <ResultPanel result={viewing} />}
      </Modal>

      <Modal open={!!deleting} onClose={() => setDeleting(null)} title="Delete prediction?">
        <p className="text-sm text-ink-2">
          This permanently removes the record for <span className="font-medium text-ink">{deleting?.filename}</span> and its stored
          images. This cannot be undone.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={() => setDeleting(null)}>Cancel</Button>
          <Button variant="danger" onClick={confirmDelete}>
            <Trash2 className="size-4" /> Delete
          </Button>
        </div>
      </Modal>
    </>
  );
}
