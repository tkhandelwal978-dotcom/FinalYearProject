import { useEffect, useState } from "react";
import { AlertCircle, RotateCcw, ScanSearch, Sparkles, X } from "lucide-react";
import Dropzone, { validateFile } from "../components/Dropzone";
import ResultPanel from "../components/ResultPanel";
import { Button, Card, PageHeader } from "../components/ui";
import { api, errorMessage } from "../services/api";
import { bytes } from "../utils/format";

export default function Detect() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [gradcam, setGradcam] = useState(true);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const selectFile = (selected) => {
    const problem = validateFile(selected);
    setResult(null);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  const analyze = async () => {
    setAnalyzing(true);
    setError(null);
    setProgress(0);
    try {
      const data = await api.predict(file, {
        gradcam,
        onUploadProgress: (e) => e.total && setProgress(Math.round((e.loaded / e.total) * 100)),
      });
      setResult(data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Image Detection"
        description="Upload an image and the trained EfficientNet-B0 model will check it for fire."
        actions={
          (file || result) && (
            <Button variant="secondary" onClick={reset} disabled={analyzing}>
              <RotateCcw className="size-4" /> Upload another image
            </Button>
          )
        }
      />

      {error && (
        <div className="mb-5 flex items-start gap-3 rounded-xl border border-[var(--fire)]/40 bg-[var(--fire)]/8 p-4 text-sm text-ink" role="alert">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-fire-ui" />
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">
            <X className="size-4 text-ink-3" />
          </button>
        </div>
      )}

      {result ? (
        <Card>
          <ResultPanel result={result} />
        </Card>
      ) : !file ? (
        <Card>
          <Dropzone onFile={selectFile} />
        </Card>
      ) : (
        <div className="grid gap-5 lg:grid-cols-5">
          <Card className="lg:col-span-3" title="Preview" subtitle={`${file.name} · ${bytes(file.size)}`}>
            <div className="relative overflow-hidden rounded-xl border border-default bg-black/5 dark:bg-black/30">
              <img src={preview} alt="Selected upload preview" className="mx-auto max-h-[440px] w-full object-contain" />
              {analyzing && (
                <div className="pointer-events-none absolute inset-0 overflow-hidden bg-[var(--ai)]/10">
                  <div className="scan-line h-1/3 w-full bg-gradient-to-b from-transparent via-[var(--ai)]/40 to-transparent" />
                </div>
              )}
            </div>
          </Card>
          <Card className="lg:col-span-2" title="Run analysis">
            <div className="space-y-5">
              <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-default p-3">
                <input
                  type="checkbox"
                  checked={gradcam}
                  onChange={(e) => setGradcam(e.target.checked)}
                  className="mt-0.5 size-4 accent-[var(--ai)]"
                  disabled={analyzing}
                />
                <span>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-ink">
                    <Sparkles className="size-4 text-[var(--ai-2)]" /> Explain with Grad-CAM
                  </span>
                  <span className="text-xs text-ink-3">Highlight the regions that drove the model's decision.</span>
                </span>
              </label>
              <Button className="w-full py-3 text-base" onClick={analyze} disabled={analyzing}>
                <ScanSearch className={`size-5 ${analyzing ? "animate-pulse" : ""}`} />
                {analyzing ? (progress < 100 ? `Uploading… ${progress}%` : "Analysing image…") : "Analyze Image"}
              </Button>
              {analyzing && (
                <div className="h-1.5 overflow-hidden rounded-full bg-black/8 dark:bg-white/8">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r from-[var(--ai)] to-[var(--ai-2)] transition-all ${progress >= 100 ? "animate-pulse" : ""}`}
                    style={{ width: `${Math.max(progress, 8)}%` }}
                  />
                </div>
              )}
              <Button variant="ghost" className="w-full" onClick={reset} disabled={analyzing}>
                Clear
              </Button>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
