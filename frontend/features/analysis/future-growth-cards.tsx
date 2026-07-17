"use client";

// Decision-first Future Growth view (Phase 15.20). The backend has no dedicated
// future-growth endpoint — run_future_agent computes the growth fields during
// enrichment and exposes them on the investment advice rows (growth_label,
// growth_reason, and when selected future_signals, infra_detected,
// growth_score). Executive-report hierarchy: Decision Summary (the verbatim
// backend growth_label) → Why checklist built from the verbatim infrastructure
// and future-signal lists → growth-score gauge → the verbatim growth_reason as
// the outlook. The label is shown ONCE (in the summary), not repeated across
// header pill and "Growth potential" pills as before.

import { LineChart, Sparkles, TrendingUp } from "lucide-react";

import { EmptyState } from "@/components/ui/empty-state";
import type { AdvisorRow } from "@/types/dashboard";
import { toneKey, toneRank } from "@/lib/value-tone";
import { formatScore, splitList } from "@/features/dashboard/format";
import { CompactPropertyHeader } from "./property-header";
import {
  MetricExplainer,
  RadialGauge,
  SectionLabel,
  hasValue,
} from "./ui/analysis-ui";
import {
  DecisionSummary,
  RecommendationBar,
  WhyCard,
} from "./ui/decision-summary";
import { ExecutiveSummary, usePropertyName } from "./ui/executive-summary";

function toNumber(value: unknown): number | null {
  const n = typeof value === "string" ? Number(value) : (value as number);
  return typeof n === "number" && !Number.isNaN(n) ? n : null;
}

export function FutureGrowthCards({ rows }: { rows: AdvisorRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend growth signals read strongest — the highest
  // backend growth_score when scores are present, otherwise the growth_label
  // whose wording reads best (toneRank). Presentation only.
  const contenders = rows
    .filter((item) => hasValue(item.growth_label))
    .map((item) => ({
      item,
      label: String(item.growth_label),
      score: toNumber(item.growth_score),
    }));
  const useScores =
    contenders.filter((c) => c.score !== null).length >= 2;
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) =>
          (useScores
            ? (c.score ?? -1) > (best.score ?? -1)
            : toneRank(c.label) > toneRank(best.label))
            ? c
            : best
        )
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Better growth potential"
            id={winner.item.id}
            name={resolveName(winner.item.id)}
            badge={winner.label}
            statement={
              hasValue(winner.item.growth_reason)
                ? String(winner.item.growth_reason)
                : undefined
            }
            stat={
              hasValue(winner.item.growth_score)
                ? {
                    label: "Growth score",
                    value: formatScore(winner.item.growth_score as number),
                    sub: "Out of 5",
                  }
                : undefined
            }
            reasons={[
              ...splitList(winner.item.infra_detected as string | null),
              ...splitList(winner.item.future_signals as string | null),
            ].slice(0, 5)}
            contenders={contenders.map((c) => ({
              id: c.item.id,
              name: resolveName(c.item.id),
              status: c.label,
              display:
                c.score !== null ? `Score ${formatScore(c.score)}` : undefined,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((item, index) => {
        // Growth data present for this property? growth_label is the primary
        // signal produced by the backend future agent.
        const hasGrowth =
          hasValue(item.growth_label) ||
          hasValue(item.growth_reason) ||
          hasValue(item.future_signals) ||
          hasValue(item.infra_detected) ||
          hasValue(item.growth_score);

        // Infrastructure / signal strings are comma-separated backend
        // summaries; splitting them into checklist lines is formatting only.
        const infraReasons = splitList(item.infra_detected as string | null);
        const signalReasons = splitList(item.future_signals as string | null);
        const growthScore = toNumber(item.growth_score);
        const label = hasValue(item.growth_label)
          ? String(item.growth_label)
          : null;

        return (
          <section key={item.id} className="space-y-3">
            <CompactPropertyHeader
              id={item.id}
              index={index}
              analysisLabel="Future Growth"
              icon={LineChart}
            />

            {hasGrowth ? (
              <>
                {/* 1 — the answer first: the backend's own growth label */}
                {label && (
                  <DecisionSummary
                    eyebrow="Growth potential"
                    headline={label}
                    icon={TrendingUp}
                    tone={toneKey(label)}
                    tagline="Long-term appreciation outlook"
                    stat={
                      hasValue(item.growth_score)
                        ? {
                            label: "Growth score",
                            value: formatScore(item.growth_score as number),
                            sub: "Out of 5",
                          }
                        : undefined
                    }
                  />
                )}

                {/* 2 — why: the backend's detected infrastructure & signals */}
                {label && (
                  <WhyCard
                    title={`Why ${label}?`}
                    reasons={[...infraReasons, ...signalReasons]}
                  />
                )}

                {/* 3 — metrics: growth-score gauge (arc scaled against the
                    backend's range — comparison_agent treats >= 3 as high
                    growth; 5 caps the visual; the verbatim score shows). */}
                {growthScore !== null && (
                  <div className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3 shadow-float">
                    <RadialGauge
                      value={growthScore}
                      max={5}
                      display={formatScore(item.growth_score as number)}
                    />
                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Growth score
                    </p>
                  </div>
                )}

                {/* 4 — the backend's own outlook for the area */}
                {hasValue(item.growth_reason) && (
                  <RecommendationBar
                    tone={label ? toneKey(label) : "positive"}
                    icon={LineChart}
                    title="Growth outlook"
                  >
                    {item.growth_reason}
                  </RecommendationBar>
                )}
              </>
            ) : (
              <div className="rounded-xl border bg-card px-4 py-3 shadow-float">
                <EmptyState
                  icon={Sparkles}
                  description="No future growth information is available for this property."
                  className="py-4"
                />
              </div>
            )}
          </section>
        );
      })}

      <MetricExplainer
        items={[
          {
            term: "Growth potential",
            meaning:
              "The backend's long-term appreciation outlook for the area, derived from detected infrastructure projects and development signals.",
          },
          {
            term: "Infrastructure & signals",
            meaning:
              "Concrete development drivers — metro lines, highways, IT corridors and similar projects — that historically lift property values nearby.",
          },
          {
            term: "How to use it",
            meaning:
              "Growth signals matter most for longer investment horizons. A high-growth area can justify a fair (or slightly high) price today.",
          },
        ]}
      />
    </div>
  );
}
