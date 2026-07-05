// Small display helpers for the dashboard. Formatting only — no business logic.

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

// Render an arbitrary cell value coming from an open backend record.
export function formatCell(value: string | number | null): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}
