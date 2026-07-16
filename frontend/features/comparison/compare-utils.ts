// Pure presentation helpers for the Property Comparison workspace (Phase
// 17.0). Smart cell highlighting only COMPARES existing backend values — the
// same Phase 15.21 rule the Executive Comparison summaries follow: the largest
// backend yield / margin / discount, the lowest backend price, or the backend's
// own status wording read through the shared toneRank. No new score, judgement
// or metric is ever computed here.

import { toneRank } from "@/lib/value-tone";

// Visual bucket for one matrix cell: the best backend value in a row gets the
// soft emerald tint, the worst the soft rose tint, everything else neutral.
export type CellTone = "best" | "worst" | "neutral";

export type Direction = "highest" | "lowest";

// A backend value is comparable when it is a non-empty scalar.
export function hasText(value: unknown): value is string | number {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return typeof value === "number" && !Number.isNaN(value);
}

// First number inside a backend value: 2.5, "₹1.10 Cr" → 1.1, "6.67%" → 6.67,
// "5-8%" → 5. Parsing only — the verbatim backend string is what is displayed.
export function parseNumeric(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isNaN(value) ? null : value;
  const match = String(value).replace(/,/g, "").match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;
  const n = Number(match[0]);
  return Number.isNaN(n) ? null : n;
}

// Mark the extreme values of a row. Rows with fewer than two comparable values,
// or where every value is equal, stay entirely neutral (nothing to highlight).
function extremes(
  values: (number | null)[],
  direction: Direction
): CellTone[] {
  const present = values.filter((v): v is number => v !== null);
  if (present.length < 2) return values.map(() => "neutral");

  const best =
    direction === "highest" ? Math.max(...present) : Math.min(...present);
  const worst =
    direction === "highest" ? Math.min(...present) : Math.max(...present);
  if (best === worst) return values.map(() => "neutral");

  return values.map((v) => {
    if (v === null) return "neutral";
    if (v === best) return "best";
    if (v === worst) return "worst";
    return "neutral";
  });
}

// Numeric row: the backend value in the winning direction is "best".
export function rankNumericCells(
  values: unknown[],
  direction: Direction
): CellTone[] {
  return extremes(values.map(parseNumeric), direction);
}

// Status row (risk_label, growth_label, verdict, rating, flag, power): rank
// the backend's own wording through the shared toneRank, so "Low Risk" beats
// "Balanced" beats "High Risk" exactly as the pills color them.
export function rankStatusCells(values: unknown[]): CellTone[] {
  return extremes(
    values.map((v) => (hasText(v) ? toneRank(String(v)) : null)),
    "highest"
  );
}

// True when every compared property carries the SAME backend value for a row
// (e.g. Bedrooms 2 / 2 / 2). The "Show Only Differences" toggle (Phase 17.2)
// hides these rows so the comparison reads faster; a missing value counts as
// a difference.
export function allIdentical(values: unknown[]): boolean {
  if (values.length < 2) return false;
  if (!values.every(hasText)) return false;
  const first = String(values[0]).trim();
  return values.every((value) => String(value).trim() === first);
}

// Proportional bar widths for a numeric row (Phase 17.1, Goal 2): the best
// backend value gets the fullest bar, the worst a small stub, everything in
// between scales linearly. Display-only scaling of existing backend numbers —
// the verbatim value is still what is shown next to the bar. Rows with fewer
// than two comparable values (or all equal) get no bars.
export function barFractions(
  values: unknown[],
  direction: Direction
): (number | null)[] {
  const nums = values.map(parseNumeric);
  const present = nums.filter((v): v is number => v !== null);
  if (present.length < 2) return nums.map(() => null);
  const max = Math.max(...present);
  const min = Math.min(...present);
  if (max === min) return nums.map(() => null);
  return nums.map((v) => {
    if (v === null) return null;
    // Normalize into [0, 1] within the row, oriented so the winning direction
    // fills the bar; the floor keeps the worst bar visible.
    const t = (v - min) / (max - min);
    const oriented = direction === "highest" ? t : 1 - t;
    return 0.15 + oriented * 0.85;
  });
}

// Relative-difference note for a numeric row (Phase 17.1, Goal 6): the best
// backend value gets one helper line comparing it against the worst — e.g.
// "₹1.05 Cr cheaper than Ariha Signature". Simple display subtraction of two
// existing backend values; every other cell stays note-free.
export function relativeNotes(
  values: unknown[],
  direction: Direction,
  names: string[],
  note: (diff: number, otherName: string) => string
): (string | null)[] {
  const nums = values.map(parseNumeric);
  const present = nums
    .map((v, i) => (v === null ? null : { v, i }))
    .filter((x): x is { v: number; i: number } => x !== null);
  if (present.length < 2) return nums.map(() => null);

  const byBest = [...present].sort((a, b) =>
    direction === "highest" ? b.v - a.v : a.v - b.v
  );
  const best = byBest[0];
  const worst = byBest[byBest.length - 1];
  if (best.v === worst.v) return nums.map(() => null);

  return nums.map((_, i) =>
    i === best.i ? note(Math.abs(best.v - worst.v), names[worst.i] ?? "") : null
  );
}

// Level row (demand_level and similar High/Medium/Low backend gradings):
// ordinal reading of the backend's own scale — presentation only.
const LEVEL_RANKS: [RegExp, number][] = [
  [/very high|high/, 3],
  [/medium|moderate|average/, 2],
  [/low/, 1],
];

export function rankLevelCells(values: unknown[]): CellTone[] {
  return extremes(
    values.map((v) => {
      if (!hasText(v)) return null;
      const text = String(v).toLowerCase();
      const match = LEVEL_RANKS.find(([pattern]) => pattern.test(text));
      return match ? match[1] : null;
    }),
    "highest"
  );
}
