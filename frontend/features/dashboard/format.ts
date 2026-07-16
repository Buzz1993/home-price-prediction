// Small display helpers for the dashboard. Formatting only — no business logic.

// Shared numeric formatters (Phase 17.5). Backend values arrive with raw
// floating-point precision (e.g. 0.7000000000000001) — never render them
// directly; route every score/percent/count through these instead.

// Scores (0–1 or 0–10 style ratings). Always two decimals: 0.7 -> "0.70".
export function formatScore(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(2);
}

// Percentages. Always two decimals: 6.666666 -> "6.67%", 46.000000001 ->
// "46.00%", 17.2 -> "17.20%". Accepts backend strings that already carry a
// "%" suffix (e.g. "5%").
export function formatPercent(value: number | string | null | undefined): string {
  const n =
    typeof value === "string" ? Number(value.replace(/%\s*$/, "").trim()) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toFixed(2)}%`;
}

// Counts (wins, properties, concerns). Always whole numbers.
export function formatInteger(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return String(Math.round(n));
}

// Currency in ₹ with Indian digit grouping: 733920 -> "₹7,33,920".
export function formatCurrency(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

// Generic number cleanup for values of unknown kind (open backend records):
// caps floats at two decimals and trims trailing zeros, leaves integers alone.
export function formatNumber(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Number.isInteger(n)) return String(n);
  return trimZeros(n.toFixed(2));
}

function trimZeros(s: string): string {
  return s.replace(/\.?0+$/, "");
}

// Prices from the backend are in Crores (₹). Render them consistently.
export function formatCr(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `₹${n.toFixed(2)} Cr`;
}

// Turn a snake_case backend column into a readable header (e.g. "monthly_rent"
// -> "Monthly Rent").
export function humanizeKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Split a backend comma-separated string (amenities / features) into a trimmed,
// non-empty list for rendering as badges.
export function splitList(value: string | null | undefined): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

// Render an arbitrary cell value coming from an open backend record. Numbers
// are cleaned up through formatNumber so raw floating-point precision
// (0.7000000000000001) never reaches the UI; other strings pass through.
export function formatCell(value: string | number | null): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  // Numeric strings with float noise (e.g. "0.39999999999999997") get the
  // same cleanup; identifiers and text are left untouched.
  if (/^-?\d+\.\d{3,}$/.test(value)) return formatNumber(value);
  return String(value);
}

// Area in square feet from the backend. Render with a thousands separator.
export function formatArea(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toLocaleString("en-IN")} sq.ft.`;
}

// Cost per square foot (₹) from the backend.
export function formatPerSqft(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `₹${Math.round(n).toLocaleString("en-IN")}/sq.ft.`;
}

// Compact price label for map markers. Prices from the backend are in Crores;
// values below 1 Cr are shown in Lakh. Trailing zeros are dropped so 2.00 -> "2
// Cr", 1.25 -> "1.25 Cr", 0.50 -> "50 Lakh". Formatting only — no business logic.
export function formatPriceLabel(value: number | string | null | undefined): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const trim = (s: string) => s.replace(/\.?0+$/, "");
  if (n >= 1) return `₹${trim(n.toFixed(2))} Cr`;
  return `₹${trim((n * 100).toFixed(2))} Lakh`;
}
