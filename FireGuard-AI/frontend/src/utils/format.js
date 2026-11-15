export const pct = (value, digits = 1) => (value == null ? "—" : `${(value * 100).toFixed(digits)}%`);

/** Model probabilities never truly reach 0 or 1, so don't let rounding claim they do. */
export function prob(value, digits = 2) {
  if (value == null) return "—";
  const limit = 10 ** -(digits + 2);
  if (value >= 1 - limit / 2) return `> ${(100 - 10 ** -digits).toFixed(digits)}%`;
  if (value <= limit / 2) return `< ${(10 ** -digits).toFixed(digits)}%`;
  return pct(value, digits);
}

export const num = (value) => (value == null ? "—" : Number(value).toLocaleString());

/** Backend timestamps are naive UTC; mark them as UTC before formatting in local time. */
export function toDate(value) {
  if (!value) return null;
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value}Z`);
}

export const dateTime = (value) =>
  toDate(value)?.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) ?? "—";

export const shortDay = (isoDay) =>
  new Date(`${isoDay}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

export const bytes = (n) => (n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);

export const CLASS_META = {
  fire: { label: "Fire Detected", short: "Fire", color: "var(--fire)" },
  non_fire: { label: "No Fire Detected", short: "No fire", color: "var(--safe)" },
};
