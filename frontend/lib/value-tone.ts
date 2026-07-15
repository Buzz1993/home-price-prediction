// Shared status-tone helper (Phase 15.18, extracted verbatim from
// report-document.tsx Phase 15.17). Tints a backend status value using only the
// value's own wording — no new judgement is ever made here. Reused by the
// report document badges and the premium analysis renderers so the whole app
// colors the same wording the same way.

// Coarse tone bucket for a backend status value (Phase 15.20). Single source
// of the wording → tone rules, shared by the pills below and the Decision
// Summary cards so a given backend wording is always colored the same way.
export type ToneKey = "positive" | "negative" | "warning" | "neutral";

export function toneKey(value: string): ToneKey {
  const v = value.toLowerCase();
  if (/🔴|overpriced|high risk|risky|avoid/.test(v)) return "negative";
  if (/🟢|undervalued|low risk|high growth|excellent|good|strong/.test(v))
    return "positive";
  if (/🟡|fair|medium|moderate|mature|caution/.test(v)) return "warning";
  return "neutral";
}

// Ordinal reading of a tone for cross-property comparisons (Phase 15.21):
// which backend wording reads better. Positive wording (undervalued, low
// risk, excellent) outranks cautionary wording, which outranks negative
// wording. Comparing two existing backend statuses is presentation only —
// no new score is computed.
const TONE_RANK: Record<ToneKey, number> = {
  positive: 3,
  warning: 2,
  neutral: 1,
  negative: 0,
};

export function toneRank(value: string): number {
  return TONE_RANK[toneKey(value)];
}

const PILL_CLASSES: Record<ToneKey, string> = {
  negative: "bg-red-50 text-red-700",
  positive: "bg-primary/10 text-primary",
  warning: "bg-amber-50 text-amber-700",
  neutral: "bg-muted text-foreground",
};

export function valueTone(value: string): string {
  return PILL_CLASSES[toneKey(value)];
}
