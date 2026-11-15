import { useRef, useState } from "react";
import { ImageUp } from "lucide-react";
import { Button } from "./ui";

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ACCEPTED_EXT = /\.(jpe?g|png|webp)$/i;
export const MAX_MB = 10;

/** Client-side checks mirror the backend's; the server re-validates everything regardless. */
export function validateFile(file) {
  if (!file) return "No file selected.";
  if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED_EXT.test(file.name))
    return "Unsupported file type. Please upload a JPG, JPEG, PNG or WEBP image.";
  if (file.size === 0) return "This file is empty.";
  if (file.size > MAX_MB * 1024 * 1024) return `File is larger than ${MAX_MB} MB.`;
  return null;
}

export default function Dropzone({ onFile, disabled }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (!disabled && files?.length) onFile(files[0]);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      className={`flex min-h-[320px] cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-8 text-center transition ${
        dragging
          ? "scale-[1.01] border-[var(--ai)] bg-[var(--ai)]/8"
          : "border-default hover:border-[var(--ai)]/60 hover:bg-[var(--ai)]/4"
      }`}
    >
      <span className="grid size-16 place-items-center rounded-2xl bg-gradient-to-br from-[var(--ai)] to-[var(--ai-2)] text-white shadow-lg">
        <ImageUp className="size-8" />
      </span>
      <div>
        <p className="text-lg font-semibold text-ink">{dragging ? "Drop to upload" : "Drag & drop an image here"}</p>
        <p className="mt-1 text-sm text-ink-3">JPG, JPEG, PNG or WEBP · up to {MAX_MB} MB</p>
      </div>
      <Button
        type="button"
        variant="secondary"
        onClick={(e) => {
          e.stopPropagation();
          input.current?.click();
        }}
      >
        Browse files
      </Button>
      <input
        ref={input}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
