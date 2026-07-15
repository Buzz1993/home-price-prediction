"use client";

// Decision-first shared primitives (Phase 15.20). Every analysis renderer now
// follows the same executive-report hierarchy: DecisionSummary (the answer,
// readable in 3 seconds) → WhyCard (reasons built from backend values) →
// metrics → RecommendationBar (what to do) → TechnicalDetails (collapsed
// extras). Pure presentation: headlines, reasons and actions are derived ONLY
// from existing backend response fields — the sign of a backend difference,
// verbatim status wording, verbatim strategy/assessment texts. Nothing is
// predicted, scored or judged here.

import { useState } from "react";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  Star,
  type LucideIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { toneKey, type ToneKey } from "@/lib/value-tone";
import { CalloutCard } from "./analysis-ui";

export { toneKey, type ToneKey };

// Card-level styling per tone — same wording rules as the status pills
// (lib/value-tone.ts), scaled up to the summary card.
const SUMMARY_TONES: Record<
  ToneKey,
  { bar: string; headline: string; card: string }
> = {
  positive: {
    bar: "bg-primary",
    headline: "text-primary",
    card: "border-primary/30 bg-primary/5",
  },
  negative: {
    bar: "bg-red-500",
    headline: "text-red-600",
    card: "border-red-200 bg-red-50/50",
  },
  warning: {
    bar: "bg-amber-500",
    headline: "text-amber-700",
    card: "border-amber-200 bg-amber-50/40",
  },
  neutral: {
    bar: "bg-border",
    headline: "text-foreground",
    card: "bg-card",
  },
};

// Map a backend rating/verdict WORD to a star count — presentation of the
// backend's own grading scale, not a new score. Returns null for wording that
// doesn't match a known grade so no stars are invented.
export function ratingStars(rating: string): number | null {
  const v = rating.toLowerCase();
  if (v.includes("excellent")) return 5;
  if (v.includes("very good") || v.includes("good")) return 4;
  if (v.includes("moderate") || v.includes("average") || v.includes("fair"))
    return 3;
  if (v.includes("poor") || v.includes("weak") || v.includes("below")) return 2;
  if (v.includes("avoid") || v.includes("bad")) return 1;
  return null;
}

// Star restatement of a backend grade. `onDark` adjusts the empty-star color
// for the green Executive Summary banner (Phase 15.21).
export function StarRow({
  count,
  onDark = false,
}: {
  count: number;
  onDark?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-0.5"
      aria-label={`${count} out of 5 stars`}
    >
      {Array.from({ length: 5 }, (_, i) => (
        <Star
          key={i}
          className={cn(
            "size-4",
            i < count
              ? onDark
                ? "fill-amber-300 text-amber-300"
                : "fill-amber-400 text-amber-400"
              : onDark
                ? "fill-primary-foreground/20 text-primary-foreground/20"
                : "fill-muted text-muted"
          )}
        />
      ))}
    </span>
  );
}

// ---------------------------------------------------------------------------
// DecisionSummary — the final result, first. Large tone-tinted headline (the
// verbatim/derived backend status), an action tagline, a small detail line and
// an optional prominent stat (e.g. the predicted value or target offer).
// ---------------------------------------------------------------------------
export function DecisionSummary({
  eyebrow,
  headline,
  icon: Icon,
  tone,
  stars,
  tagline,
  detail,
  stat,
}: {
  eyebrow: string;
  headline: string;
  icon?: LucideIcon;
  // Defaults to the tone of the headline's own wording.
  tone?: ToneKey;
  stars?: number | null;
  tagline?: string;
  detail?: string;
  stat?: { label: string; value: string; sub?: string };
}) {
  const styles = SUMMARY_TONES[tone ?? toneKey(headline)];
  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border shadow-sm",
        styles.card
      )}
    >
      <div className={cn("h-1", styles.bar)} />
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 p-4">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
            {eyebrow}
          </p>
          <p
            className={cn(
              "mt-0.5 flex items-center gap-2 font-heading text-2xl font-bold uppercase tracking-tight sm:text-3xl",
              styles.headline
            )}
          >
            {Icon && <Icon className="size-6 shrink-0 sm:size-7" />}
            <span className="min-w-0 break-words">{headline}</span>
          </p>
          {typeof stars === "number" && (
            <div className="mt-1">
              <StarRow count={stars} />
            </div>
          )}
          {tagline && (
            <p className="mt-1 text-sm font-medium text-foreground">
              {tagline}
            </p>
          )}
          {detail && (
            <p className="mt-0.5 text-xs text-muted-foreground">{detail}</p>
          )}
        </div>

        {stat && (
          <div className="shrink-0 text-right">
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {stat.label}
            </p>
            <p
              className={cn(
                "font-heading text-2xl font-bold tabular-nums sm:text-3xl",
                styles.headline
              )}
            >
              {stat.value}
            </p>
            {stat.sub && (
              <p className="text-[11px] text-muted-foreground">{stat.sub}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// WhyCard — "Why this result?" checklist. Each reason is a short line the
// caller assembled from backend fields (yields, differences, chips, verbatim
// lists) — never generic filler. Hidden when there are no reasons.
// ---------------------------------------------------------------------------
export function WhyCard({
  title,
  reasons,
}: {
  title: string;
  reasons: string[];
}) {
  if (reasons.length === 0) return null;
  return (
    <div className="rounded-xl border bg-card p-3 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <ul className="mt-2 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        {reasons.map((reason, i) => (
          <li key={i} className="flex items-start gap-2 text-sm leading-snug">
            <Check className="mt-0.5 size-4 shrink-0 text-primary" />
            <span className="min-w-0 break-words">{reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RecommendationBar — the closing "what should you do" strip. Reuses the
// existing CalloutCard; the text is a verbatim backend strategy/assessment or
// an action phrase derived from a backend value's sign/wording.
// ---------------------------------------------------------------------------
const CALLOUT_TONE: Record<ToneKey, "green" | "red" | "amber" | "blue"> = {
  positive: "green",
  negative: "red",
  warning: "amber",
  neutral: "blue",
};

export function RecommendationBar({
  tone = "positive",
  icon = CheckCircle2,
  title = "Recommended action",
  children,
}: {
  tone?: ToneKey;
  icon?: LucideIcon;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <CalloutCard tone={CALLOUT_TONE[tone]} icon={icon} title={title}>
      <span className="font-medium">{children}</span>
    </CalloutCard>
  );
}

// ---------------------------------------------------------------------------
// TechnicalDetails — collapsed container for the long tail (extra backend
// fields, raw records). Keeps every field reachable without competing with
// the decision. Same collapsible pattern as MetricExplainer.
// ---------------------------------------------------------------------------
export function TechnicalDetails({
  title = "Technical details",
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border bg-muted/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        {title}
        <ChevronDown
          className={cn(
            "size-4 shrink-0 transition-transform",
            open && "rotate-180"
          )}
        />
      </button>
      {open && <div className="border-t px-3 pb-3 pt-2">{children}</div>}
    </div>
  );
}
