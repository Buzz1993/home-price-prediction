"use client";

// Shared premium UI primitives for the AI Analysis workspace (Phase 15.18).
// Pure presentation: these components render backend values verbatim in the
// report-document design language (Phase 15.17) — compact dashboard cards with
// soft shadows, uppercase micro-labels, dashed key-value rows and tone-tinted
// status pills. Nothing is computed here beyond formatting and a percentage
// for gauges/bars; every displayed value comes from the backend response.

import { useState } from "react";
import { ChevronDown, type LucideIcon } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { valueTone } from "@/lib/value-tone";
import {
  formatCell,
  formatCr,
  formatNumber,
  humanizeKey,
} from "@/features/dashboard/format";

// A backend field is "present" only when it is a non-empty value.
export function hasValue(value: unknown): value is string | number {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

// ---------------------------------------------------------------------------
// StatusPill — rounded pill tinted from the value's own wording (valueTone).
// ---------------------------------------------------------------------------
export function StatusPill({
  value,
  className,
}: {
  value: string | number;
  className?: string;
}) {
  const text = String(value);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        valueTone(text),
        className
      )}
    >
      {text}
    </span>
  );
}

// ---------------------------------------------------------------------------
// MetricCard — compact stat tile: icon + uppercase micro-label, large value,
// optional muted sub-line. `tone` colors the value (e.g. green gain / red
// loss); the caller derives it only from backend values (like the sign of
// margin_diff), never new judgement.
// ---------------------------------------------------------------------------
export function MetricCard({
  label,
  value,
  sub,
  tone,
  highlight,
  icon: Icon,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "positive" | "negative" | "neutral";
  highlight?: boolean;
  icon?: LucideIcon;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border bg-card px-3 py-2.5 shadow-sm transition-shadow hover:shadow-md",
        highlight && "border-primary/40 bg-primary/5"
      )}
    >
      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {Icon && <Icon className="size-3.5 shrink-0 text-primary/70" />}
        {label}
      </p>
      <p
        className={cn(
          "mt-0.5 truncate font-heading text-lg font-semibold tabular-nums",
          tone === "positive" && "text-primary",
          tone === "negative" && "text-red-600"
        )}
      >
        {value}
      </p>
      {sub && (
        <p className="truncate text-[11px] text-muted-foreground">{sub}</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// PillCard — compact stat tile whose value is one or more status pills
// (used for ratings, severity, demand, negotiation power).
// ---------------------------------------------------------------------------
export function PillCard({
  label,
  values,
  icon: Icon,
}: {
  label: string;
  values: (string | number)[];
  icon?: LucideIcon;
}) {
  return (
    <div className="rounded-xl border bg-card px-3 py-2.5 shadow-sm transition-shadow hover:shadow-md">
      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {Icon && <Icon className="size-3.5 shrink-0 text-primary/70" />}
        {label}
      </p>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {values.map((value, index) => (
          <StatusPill key={index} value={value} />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SectionLabel — small uppercase section header used above groups of cards.
// ---------------------------------------------------------------------------
export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </p>
  );
}

// ---------------------------------------------------------------------------
// KeyValueList — dashed-border dl rows for an open backend record. Renders
// EVERY remaining field so no backend field is ever ignored; known price
// columns are formatted in ₹ Cr, status-like values become pills.
// ---------------------------------------------------------------------------
const PRICE_KEYS = new Set([
  "price",
  "original_price",
  "predicted_price",
  "margin_diff",
  "target_price",
]);

// Keys whose values read as a status — rendered as a tinted pill.
const PILL_KEYS = new Set([
  "status",
  "analysis_flag",
  "analysis_severity",
  "demand_level",
  "investment_rating",
  "negotiation_power",
  "risk_label",
  "growth_label",
  "price_position",
  "verdict",
]);

export function KeyValueRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 border-b border-dashed border-border/70 pb-1 last:border-0 last:pb-0">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="text-right text-sm font-medium text-foreground">
        {value}
      </dd>
    </div>
  );
}

// Whether a backend record still carries fields beyond the ones a renderer
// already gave a dedicated treatment — used to decide if the collapsed
// "Technical details" section is needed at all.
export function hasExtraFields(
  record: Record<string, unknown>,
  omit: string[]
): boolean {
  const omitted = new Set(omit);
  return Object.entries(record).some(
    ([key, value]) => !omitted.has(key) && hasValue(value)
  );
}

export function KeyValueList({
  record,
  omit = [],
}: {
  record: Record<string, unknown>;
  omit?: string[];
}) {
  const omitted = new Set(omit);
  // The hasValue filter guarantees every remaining value is a string | number.
  const entries = Object.entries(record).filter(
    ([key, value]) => !omitted.has(key) && hasValue(value)
  ) as [string, string | number][];
  if (entries.length === 0) return null;

  return (
    <dl className="space-y-1.5 rounded-xl border bg-card px-3 py-2.5 shadow-sm">
      {entries.map(([key, value]) => (
        <KeyValueRow
          key={key}
          label={humanizeKey(key)}
          value={
            PILL_KEYS.has(key) ? (
              <StatusPill value={value} />
            ) : PRICE_KEYS.has(key) ? (
              formatCr(value as number | string)
            ) : (
              formatCell(value as string | number | null)
            )
          }
        />
      ))}
    </dl>
  );
}

// ---------------------------------------------------------------------------
// RadialGauge — compact progress ring for an existing backend score. The only
// computation is the arc length (value / max); the verbatim backend text is
// what is displayed in the center. Tone follows the same wording rules as
// pills (the caller passes it from backend values only).
// ---------------------------------------------------------------------------
export function RadialGauge({
  value,
  max,
  display,
  tone = "primary",
}: {
  value: number;
  max: number;
  // Verbatim text shown inside the ring (defaults to the raw number).
  display?: string;
  tone?: "primary" | "red" | "amber";
}) {
  const percent = Math.max(0, Math.min(1, value / max));
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  return (
    <div className="relative size-16 shrink-0">
      <svg viewBox="0 0 64 64" className="size-full -rotate-90">
        <circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          strokeWidth="6"
          className="stroke-muted"
        />
        <circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - percent)}
          className={cn(
            "transition-all duration-500",
            tone === "primary" && "stroke-primary",
            tone === "red" && "stroke-red-500",
            tone === "amber" && "stroke-amber-500"
          )}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold tabular-nums">
        {display ?? formatNumber(value)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MiniCompareBars — compact side-by-side value comparison (label + verbatim
// value per row with a thin proportional bar). Replaces the former full-width
// score bars; widths are proportional to the backend values (UI only).
// ---------------------------------------------------------------------------
export function MiniCompareBars({
  items,
}: {
  items: { label: string; value: number; display: string; emphasis?: boolean }[];
}) {
  const max = Math.max(...items.map((item) => item.value), 0);
  if (max <= 0) return null;
  return (
    <div className="space-y-1.5">
      {items.map((item) => (
        <div key={item.label}>
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[11px] font-medium text-muted-foreground">
              {item.label}
            </span>
            <span className="text-xs font-semibold tabular-nums">
              {item.display}
            </span>
          </div>
          <div className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                item.emphasis ? "bg-primary" : "bg-muted-foreground/40"
              )}
              style={{ width: `${(item.value / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InfoChip — small rounded chip for signals / infrastructure / tags.
// ---------------------------------------------------------------------------
export function InfoChip({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-primary/20 bg-primary/5 px-2 py-0.5 text-xs font-medium text-foreground">
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// CalloutCard — tinted card for recommendations, concerns and summaries.
// ---------------------------------------------------------------------------
const CALLOUT_TONES = {
  green: {
    card: "border-primary/30 bg-primary/5",
    title: "text-primary",
    icon: "text-primary",
  },
  red: {
    card: "border-red-200 bg-red-50/60",
    title: "text-red-700",
    icon: "text-red-600",
  },
  amber: {
    card: "border-amber-200 bg-amber-50/60",
    title: "text-amber-700",
    icon: "text-amber-600",
  },
  blue: {
    card: "border-blue-200 bg-blue-50/60",
    title: "text-blue-700",
    icon: "text-blue-600",
  },
} as const;

export function CalloutCard({
  tone,
  icon: Icon,
  title,
  children,
}: {
  tone: keyof typeof CALLOUT_TONES;
  icon?: LucideIcon;
  title: string;
  children: React.ReactNode;
}) {
  const styles = CALLOUT_TONES[tone];
  return (
    <div className={cn("rounded-xl border p-3 shadow-sm", styles.card)}>
      <p
        className={cn(
          "mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide",
          styles.title
        )}
      >
        {Icon && <Icon className={cn("size-3.5", styles.icon)} />}
        {title}
      </p>
      <div className="text-sm leading-relaxed text-foreground">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricExplainer — collapsible "What does this mean?" footer with STATIC
// educational copy about the metric itself. It never describes the data —
// only how investors should read this kind of analysis.
// ---------------------------------------------------------------------------
export function MetricExplainer({
  items,
}: {
  items: { term: string; meaning: string }[];
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border bg-muted/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        What do these metrics mean?
        <ChevronDown
          className={cn("size-4 shrink-0 transition-transform", open && "rotate-180")}
        />
      </button>
      {open && (
        <dl className="space-y-2 border-t px-3 pb-3 pt-2">
          {items.map((item) => (
            <div key={item.term}>
              <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {item.term}
              </dt>
              <dd className="text-sm leading-relaxed text-foreground">
                {item.meaning}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AnalysisSkeleton — compact loading placeholder: header bar + metric grid.
// ---------------------------------------------------------------------------
export function AnalysisSkeleton() {
  return (
    <div className="space-y-2" aria-busy="true" aria-label="Loading analysis">
      <Skeleton className="h-20 rounded-xl" />
      <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
        <Skeleton className="h-[4.5rem] rounded-xl" />
        <Skeleton className="h-[4.5rem] rounded-xl" />
        <Skeleton className="h-[4.5rem] rounded-xl" />
        <Skeleton className="h-[4.5rem] rounded-xl" />
      </div>
      <Skeleton className="h-24 rounded-xl" />
    </div>
  );
}
