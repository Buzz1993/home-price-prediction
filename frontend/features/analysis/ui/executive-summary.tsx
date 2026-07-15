"use client";

// Executive Comparison Summary (Phase 15.21). When several properties are
// analyzed together, every analysis now opens with this card — "which property
// performs better in this category?" — before the individual per-property
// sections (which keep their full Phase 15.20 decision-first layout).
//
// Pure presentation in the Premium Report design language: a solid green
// winner banner (trophy eyebrow, large property name, verbatim status badge,
// one-line explanation, key stat, ✓ reasons) over a side-by-side contender
// strip. The WINNER IS NEVER COMPUTED HERE — each renderer picks it by
// comparing existing backend values (the largest backend yield / margin /
// discount / score, or the backend's own status wording via toneRank) and
// passes the result in. No new metrics, no new calculations.

import { Check, Trophy } from "lucide-react";

import { cn } from "@/lib/utils";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { StatusPill, hasValue } from "./analysis-ui";
import { StarRow } from "./decision-summary";

// Resolve a property id to its human-readable name from the workspace's
// already-loaded backend search rows (same lookup CompactPropertyHeader and
// the comparison winner strip use). Falls back to the raw id.
export function usePropertyName(): (id: string) => string {
  const { properties } = useWorkspace();
  return (id: string) =>
    properties.find((p) => p.id === id)?.project_name || id;
}

// One property in the contender strip. `status` and `display` are verbatim
// backend values (or the same derived wording the property's own Decision
// Summary shows).
export type Contender = {
  id: string;
  name: string;
  status?: string | number;
  display?: string;
  isWinner: boolean;
};

export function ExecutiveSummary({
  eyebrow,
  name,
  id,
  badge,
  stars,
  statement,
  stat,
  reasons = [],
  contenders,
}: {
  // e.g. "Better rental investment", "Lower-risk property".
  eyebrow: string;
  name: string;
  id: string;
  // Verbatim winner status (rating / verdict / flag / power).
  badge?: string | number;
  stars?: number | null;
  // One-line explanation built from the winner's backend values.
  statement?: string;
  stat?: { label: string; value: string; sub?: string };
  // ✓ reasons, verbatim backend list items from the winner's row.
  reasons?: string[];
  contenders?: Contender[];
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-primary/30 shadow-sm">
      {/* Winner banner — Premium Report green */}
      <div className="bg-primary p-4 text-primary-foreground">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
          <Trophy className="size-3.5" /> {eyebrow}
        </p>

        <div className="mt-1 flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
          <div className="min-w-0">
            <h3 className="break-words font-heading text-2xl font-bold leading-tight sm:text-3xl">
              {name}
            </h3>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              {hasValue(badge) && (
                <span className="rounded-full bg-primary-foreground/15 px-2.5 py-0.5 text-xs font-semibold">
                  {badge}
                </span>
              )}
              {typeof stars === "number" && (
                <StarRow count={stars} onDark />
              )}
              {name !== id && (
                <span className="font-mono text-[10px] text-primary-foreground/60">
                  {id}
                </span>
              )}
            </div>
            {statement && (
              <p className="mt-1.5 text-sm leading-relaxed text-primary-foreground/85">
                {statement}
              </p>
            )}
          </div>

          {stat && (
            <div className="shrink-0 text-right">
              <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                {stat.label}
              </p>
              <p className="font-heading text-2xl font-bold tabular-nums sm:text-3xl">
                {stat.value}
              </p>
              {stat.sub && (
                <p className="text-[11px] text-primary-foreground/70">
                  {stat.sub}
                </p>
              )}
            </div>
          )}
        </div>

        {reasons.length > 0 && (
          <ul className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1 border-t border-primary-foreground/15 pt-2.5">
            {reasons.map((reason, i) => (
              <li
                key={i}
                className="flex items-center gap-1.5 text-xs font-medium text-primary-foreground/90"
              >
                <Check className="size-3.5 shrink-0" />
                {reason}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Contender strip — every compared property side by side */}
      {contenders && contenders.length > 1 && (
        <div className="grid gap-2 bg-card p-3 sm:grid-cols-2 xl:grid-cols-3">
          {contenders.map((c) => (
            <div
              key={c.id}
              className={cn(
                "rounded-lg border px-3 py-2",
                c.isWinner ? "border-primary/40 bg-primary/5" : "bg-card"
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="min-w-0 truncate text-sm font-semibold">
                  {c.name}
                </p>
                {c.isWinner && (
                  <Trophy className="size-3.5 shrink-0 text-primary" />
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
                {hasValue(c.status) && <StatusPill value={c.status} />}
                {c.display && (
                  <span className="text-sm font-semibold tabular-nums">
                    {c.display}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
