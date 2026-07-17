"use client";

// Executive Winner Summary, Final Scoreboard and Final Recommendation for the
// Property Comparison workspace (Phase 17.0). Everything shown is the backend's
// own result: the overall winner, scores, verdicts and comparison reasons come
// verbatim from POST /analysis/comparison; the category wins are the same
// backend-value comparisons the section winners use (compare-winners); the
// closing action is the winner's own backend negotiation strategy. Claude's
// optional comparison explanation renders through the shared
// AnalysisExplanation card.

import { Award, Check, Handshake, Medal, Trophy } from "lucide-react";

import { AnalysisExplanation } from "@/features/analysis/analysis-explanation";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import {
  StarRow,
  ratingStars,
} from "@/features/analysis/ui/decision-summary";
import { ExecutiveSummary } from "@/features/analysis/ui/executive-summary";
import {
  formatCr,
  formatPercent,
  formatScore,
  splitList,
} from "@/features/dashboard/format";
import { hasText } from "./compare-utils";
import { scoreboard } from "./compare-winners";
import { winTally } from "./win-tally";
import type { CompareBundle } from "./use-compare-data";

// ---------------------------------------------------------------------------
// Executive Winner Summary — the backend comparison winner, first.
// ---------------------------------------------------------------------------
export function ExecutiveWinner({
  bundle,
  resolveName,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
}) {
  const { winner, rankings } = bundle.comparison;
  const runnerUp = rankings.find((row) => row.id !== winner.id);

  return (
    <div className="space-y-3">
      <ExecutiveSummary
        eyebrow="Best overall investment"
        id={winner.id}
        name={resolveName(winner.id)}
        badge={hasText(winner.verdict) ? winner.verdict : undefined}
        stars={hasText(winner.verdict) ? ratingStars(String(winner.verdict)) : undefined}
        statement={`Ranked first of ${rankings.length} compared properties by the comparison engine`}
        stat={
          // The backend score can be null (NaN-scrubbed) — omit the stat then.
          hasText(winner.overall_score)
            ? {
                label: "Overall score",
                value: formatScore(winner.overall_score),
                sub: "Backend comparison score",
              }
            : undefined
        }
        reasons={splitList(winner.comparison_reason).slice(0, 6)}
        contenders={rankings.map((row) => ({
          id: row.id,
          name: resolveName(row.id),
          status: hasText(row.verdict) ? row.verdict : undefined,
          display: hasText(row.overall_score)
            ? `Score ${formatScore(row.overall_score)}`
            : undefined,
          isWinner: row.id === winner.id,
        }))}
      />

      {runnerUp && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-xl border bg-card px-4 py-3 shadow-float transition-shadow duration-200 hover:shadow-float-lg">
          <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            <span className="flex size-6 items-center justify-center rounded-md bg-amber-100/80">
              <Medal className="size-3.5 text-amber-500" />
            </span>
            Runner Up
          </p>
          <p className="font-heading text-sm font-semibold">
            {resolveName(runnerUp.id)}
          </p>
          {hasText(runnerUp.verdict) && <StatusPill value={runnerUp.verdict} />}
          <span className="text-xs text-muted-foreground">
            {runnerUp.comparison_reason}
          </span>
        </div>
      )}

      <AnalysisExplanation
        explanation={bundle.comparisonExplanation}
        unavailableMessage="Property comparison is available, but the AI explanation is temporarily unavailable."
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Final Scoreboard — one category win per row, all backend-derived.
// ---------------------------------------------------------------------------
export function FinalScoreboard({
  bundle,
  resolveName,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
}) {
  const entries = scoreboard(bundle);
  if (entries.length === 0) return null;

  return (
    <section className="card-accent-top overflow-hidden rounded-xl border bg-card shadow-float transition-shadow duration-200 hover:shadow-float-lg">
      <div className="border-b p-4">
        <h3 className="flex items-center gap-2.5 font-heading text-sm font-semibold">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <Award className="size-4 text-primary" />
          </span>
          Final Scoreboard
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Category winners across the compared properties.
        </p>
      </div>
      <div className="grid gap-2.5 p-4 sm:grid-cols-2 xl:grid-cols-3">
        {entries.map(({ label, winner }) => (
          <div
            key={label}
            className="rounded-xl border bg-gradient-to-br from-muted/40 to-transparent px-3.5 py-2.5 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-float"
          >
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {label}
            </p>
            <p className="mt-1 flex items-center gap-1.5 font-heading text-sm font-semibold">
              <Trophy className="size-3.5 shrink-0 text-primary" />
              <span className="min-w-0 truncate">{resolveName(winner.id)}</span>
            </p>
            {winner.points[0] && (
              <p className="mt-0.5 truncate text-xs text-muted-foreground">
                {winner.points[0]}
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Final Decision Card (Phase 17.1) — one large closing recommendation for the
// backend winner: verdict badge + stars, the backend's own overall score, a
// "Won X of Y categories" stat counted from the existing category winners
// (win-tally — no invented confidence figure), the "Why" checklist restating
// the winner's category wins and the backend's own comparison reasons, and the
// winner's verbatim backend negotiation strategy as the recommended action.
// ---------------------------------------------------------------------------
export function FinalRecommendation({
  bundle,
  resolveName,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
}) {
  const { winner } = bundle.comparison;
  const stars = hasText(winner.verdict)
    ? ratingStars(String(winner.verdict))
    : null;

  // Category wins the recommended property earned (backend-value comparisons),
  // then the backend's own comparison reasons.
  const tally = winTally(bundle);
  const winnerTally = tally.entries.find((entry) => entry.id === winner.id);
  const categoryWins = scoreboard(bundle)
    .filter((entry) => entry.winner.id === winner.id)
    .map((entry) => entry.winner.points[0] ?? entry.label);
  const reasons = Array.from(
    new Set([...categoryWins, ...splitList(winner.comparison_reason)])
  ).slice(0, 6);

  // The winner's own negotiation row closes the recommendation.
  const negotiation = bundle.negotiation.find((row) => row.id === winner.id);

  return (
    <section className="overflow-hidden rounded-xl border border-primary/30 shadow-float transition-shadow duration-200 hover:shadow-float-lg">
      <div className="bg-brand-gradient relative p-6 text-primary-foreground sm:p-7">
        {/* Soft radial sheen, purely decorative (same as the Executive hero). */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(255,255,255,0.18),transparent_55%)]"
        />
        <div className="relative">
          <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
            <span className="flex size-6 items-center justify-center rounded-md bg-primary-foreground/15">
              <Trophy className="size-3.5" />
            </span>
            Final Recommendation
          </p>
          <div className="mt-2 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
            <div className="min-w-0">
              <h3 className="break-words font-heading text-3xl font-bold leading-tight sm:text-4xl">
                {resolveName(winner.id)}
              </h3>
              <div className="mt-2.5 flex flex-wrap items-center gap-2">
                {hasText(winner.verdict) && (
                  <span className="rounded-full bg-primary-foreground/15 px-3 py-1 text-xs font-semibold backdrop-blur-sm">
                    {winner.verdict}
                  </span>
                )}
                {typeof stars === "number" && <StarRow count={stars} onDark />}
                <span className="font-mono text-[10px] text-primary-foreground/60">
                  {winner.id}
                </span>
              </div>
            </div>
            <div className="flex shrink-0 gap-8 text-right">
              {hasText(winner.overall_score) && (
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                    Overall score
                  </p>
                  <p className="mt-0.5 font-heading text-3xl font-bold tabular-nums sm:text-4xl">
                    {formatScore(winner.overall_score)}
                  </p>
                  <p className="mt-0.5 text-[11px] text-primary-foreground/70">
                    Backend comparison score
                  </p>
                </div>
              )}
              {winnerTally && tally.totalCategories > 0 && (
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                    Category wins
                  </p>
                  <p className="mt-0.5 font-heading text-3xl font-bold tabular-nums sm:text-4xl">
                    {winnerTally.wins}
                    <span className="text-base font-semibold text-primary-foreground/70">
                      /{tally.totalCategories}
                    </span>
                  </p>
                  <p className="mt-0.5 text-[11px] text-primary-foreground/70">
                    Comparison categories won
                  </p>
                </div>
              )}
            </div>
          </div>

          {reasons.length > 0 && (
            <div className="mt-5 border-t border-primary-foreground/15 pt-4">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-primary-foreground/70">
                Why this property?
              </p>
              <ul className="mt-2 grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
                {reasons.map((reason, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-1.5 text-xs font-medium text-primary-foreground/90"
                  >
                    <Check className="mt-0.5 size-3.5 shrink-0" />
                    <span className="min-w-0 break-words">{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {negotiation &&
        (hasText(negotiation.strategy) || hasText(negotiation.target_price)) && (
          <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-2 bg-card p-4">
            <p className="flex min-w-0 items-start gap-2.5 text-sm">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                <Handshake className="size-4 text-primary" />
              </span>
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Recommended action
                </span>
                <span className="break-words font-medium">
                  {hasText(negotiation.strategy)
                    ? negotiation.strategy
                    : `Open near the target price of ${formatCr(negotiation.target_price)}.`}
                </span>
              </span>
            </p>
            {hasText(negotiation.target_price) && (
              <div className="shrink-0 text-right">
                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Target offer
                </p>
                <p className="font-heading text-xl font-bold tabular-nums text-primary">
                  {formatCr(negotiation.target_price)}
                </p>
                {hasText(negotiation.suggested_discount_percent) && (
                  <p className="text-[11px] text-muted-foreground">
                    {formatPercent(negotiation.suggested_discount_percent as string)}{" "}
                    suggested discount
                  </p>
                )}
              </div>
            )}
          </div>
        )}
    </section>
  );
}
