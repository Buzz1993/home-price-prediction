"use client";

// Decision-first Risk Analysis view (Phase 15.20). The backend has no dedicated
// risk endpoint — it embeds risk metrics inside the investment advice
// (get_investment_advice), so this view surfaces the advisor rows focused on
// the risk dimension, in the executive-report hierarchy: Decision Summary (the
// verbatim backend verdict as the risk level, with the concern COUNT — a count
// of the backend's own risks list — as the tagline) → Why checklist built from
// the verbatim positives → key concerns callout → investor-fit recommendation
// read from the verbatim suitable_for field. The verdict is shown ONCE (in the
// summary), not repeated across header pill and verdict pills as before.

import { CheckCircle2, ShieldAlert, ShieldCheck } from "lucide-react";

import type { AdvisorRow } from "@/types/dashboard";
import { toneKey, toneRank } from "@/lib/value-tone";
import { splitList } from "@/features/dashboard/format";
import { CompactPropertyHeader } from "./property-header";
import {
  CalloutCard,
  MetricExplainer,
  SectionLabel,
  hasValue,
} from "./ui/analysis-ui";
import {
  DecisionSummary,
  RecommendationBar,
  WhyCard,
} from "./ui/decision-summary";
import { ExecutiveSummary, usePropertyName } from "./ui/executive-summary";

export function RiskCards({ rows }: { rows: AdvisorRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend verdict wording reads best (toneRank); ties go
  // to the property with fewer backend-listed concerns. Presentation only —
  // both signals are the backend's own.
  const contenders = rows
    .filter((item) => hasValue(item.verdict))
    .map((item) => ({
      item,
      rank: toneRank(String(item.verdict)),
      concernCount: splitList(item.risks).length,
    }));
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) =>
          c.rank > best.rank ||
          (c.rank === best.rank && c.concernCount < best.concernCount)
            ? c
            : best
        )
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Lower-risk property"
            id={winner.item.id}
            name={resolveName(winner.item.id)}
            badge={String(winner.item.verdict)}
            statement={
              winner.concernCount === 0
                ? "No major red flags identified by the analysis engine"
                : `${winner.concernCount} concern${winner.concernCount === 1 ? "" : "s"} flagged by the analysis engine`
            }
            reasons={splitList(winner.item.positives).slice(0, 4)}
            contenders={contenders.map((c) => ({
              id: c.item.id,
              name: resolveName(c.item.id),
              status: c.item.verdict,
              display: `${c.concernCount} concern${c.concernCount === 1 ? "" : "s"}`,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((item, index) => {
        // The backend risks/positives fields are comma-separated summaries;
        // splitting them for bullets is formatting only, wording stays verbatim.
        const concerns = splitList(item.risks);
        const positives = splitList(item.positives);
        const verdict = hasValue(item.verdict) ? String(item.verdict) : null;
        const tone = verdict ? toneKey(verdict) : "neutral";

        return (
          <section key={item.id} className="space-y-3">
            <CompactPropertyHeader
              id={item.id}
              index={index}
              analysisLabel="Risk Analysis"
              icon={ShieldAlert}
            />

            {/* 1 — the answer first: the backend's own verdict as risk level */}
            {verdict && (
              <DecisionSummary
                eyebrow="Risk level"
                headline={verdict}
                icon={tone === "negative" ? ShieldAlert : ShieldCheck}
                tone={tone}
                tagline={
                  concerns.length === 0
                    ? "No major red flags identified"
                    : `${concerns.length} concern${concerns.length === 1 ? "" : "s"} identified`
                }
              />
            )}

            {/* 2 — why: the backend's own positive signals */}
            {verdict && (
              <WhyCard title={`Why ${verdict}?`} reasons={positives} />
            )}

            {/* 3 — the concerns themselves */}
            {hasValue(item.risks) && (
              <CalloutCard tone="red" icon={ShieldAlert} title="Key concerns">
                {concerns.length > 0 ? (
                  <ul className="space-y-1">
                    {concerns.map((concern, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-red-500/70" />
                        <span>{concern}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  item.risks
                )}
              </CalloutCard>
            )}

            {/* 4 — what to do: the backend's own investor fit */}
            {hasValue(item.suitable_for) && (
              <RecommendationBar tone={tone} icon={CheckCircle2}>
                Suited for {item.suitable_for} investors.
              </RecommendationBar>
            )}
          </section>
        );
      })}

      <MetricExplainer
        items={[
          {
            term: "Risk level",
            meaning:
              "The engine's overall judgement of the property once risks are weighed against its strengths.",
          },
          {
            term: "Key concerns",
            meaning:
              "Risk factors the analysis engine identified for this property — pricing, market or property-specific issues that could affect the investment.",
          },
          {
            term: "How to use it",
            meaning:
              "No property is risk-free. Use the concerns as a due-diligence checklist and weigh them against the growth and rental outlook before deciding.",
          },
        ]}
      />
    </div>
  );
}
