"use client";

// Decision-first Property Valuation view (Phase 15.20). Renders the unchanged
// backend valuation rows (run_mcp_valuation: id, project_name, price,
// costpersqft, analysis_flag, analysis_msg, analysis_severity) in the
// executive-report hierarchy: Decision Summary (the verbatim backend
// analysis_flag as the headline, with an action tagline read from that same
// wording) → Why checklist built from the price / severity fields → metric
// grid → the verbatim backend analysis_msg as the assessment → collapsed
// technical details. The flag is shown ONCE (in the summary) — the former
// header pill and segmented indicator that repeated it are gone.

import { AlertTriangle, BadgeCheck, Gauge, Ruler, Scale } from "lucide-react";

import type { AnalysisRow } from "@/types/dashboard";
import { toneKey } from "@/lib/value-tone";
import { formatCr, formatPerSqft } from "@/features/dashboard/format";
import { CompactPropertyHeader } from "./property-header";
import {
  KeyValueList,
  MetricCard,
  MetricExplainer,
  PillCard,
  SectionLabel,
  hasExtraFields,
  hasValue,
} from "./ui/analysis-ui";
import { ExecutiveSummary, usePropertyName } from "./ui/executive-summary";
import {
  DecisionSummary,
  RecommendationBar,
  TechnicalDetails,
  WhyCard,
} from "./ui/decision-summary";

const KNOWN_FIELDS = [
  "id",
  "project_name",
  "price",
  "costpersqft",
  "analysis_flag",
  "analysis_msg",
  "analysis_severity",
];

// Action tagline + icon read from the backend flag's own wording — the same
// three stops the former segmented indicator highlighted.
function readFlag(flag: string) {
  const v = flag.toLowerCase();
  if (v.includes("overpriced"))
    return { tagline: "Negotiate before buying", icon: AlertTriangle };
  if (v.includes("undervalued"))
    return { tagline: "Strong buying opportunity", icon: BadgeCheck };
  if (v.includes("fair"))
    return { tagline: "Priced within the fair range", icon: Scale };
  return { tagline: undefined, icon: Gauge };
}

// Ordinal reading of the backend flag for cross-property comparison: the
// backend's own three stops, undervalued > fair > overpriced. Unknown wording
// is excluded rather than guessed.
function flagRank(flag: string): number {
  const v = flag.toLowerCase();
  if (v.includes("undervalued")) return 2;
  if (v.includes("fair")) return 1;
  if (v.includes("overpriced")) return 0;
  return -1;
}

export function ValuationResults({ rows }: { rows: AnalysisRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend analysis_flag reads best on the backend's own
  // scale (undervalued > fair > overpriced). Presentation only.
  const contenders = rows
    .map((row, i) => ({
      row,
      id: String(row.id ?? i),
      rank: hasValue(row.analysis_flag)
        ? flagRank(String(row.analysis_flag))
        : -1,
    }))
    .filter((c) => c.rank >= 0);
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) => (c.rank > best.rank ? c : best))
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Better valuation"
            id={winner.id}
            name={resolveName(winner.id)}
            badge={String(winner.row.analysis_flag)}
            statement={
              hasValue(winner.row.analysis_msg)
                ? String(winner.row.analysis_msg)
                : undefined
            }
            stat={
              hasValue(winner.row.price)
                ? {
                    label: "Listed price",
                    value: formatCr(winner.row.price),
                  }
                : undefined
            }
            contenders={contenders.map((c) => ({
              id: c.id,
              name: resolveName(c.id),
              status: c.row.analysis_flag ?? undefined,
              display: hasValue(c.row.price)
                ? formatCr(c.row.price)
                : undefined,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((row, index) => {
        const flag = hasValue(row.analysis_flag)
          ? String(row.analysis_flag)
          : null;
        const decision = flag ? readFlag(flag) : null;

        // Why checklist — restates the backend valuation fields verbatim.
        const reasons: string[] = [];
        if (hasValue(row.price))
          reasons.push(`Listed at ${formatCr(row.price)}`);
        if (hasValue(row.costpersqft))
          reasons.push(
            `Cost basis of ${formatPerSqft(row.costpersqft as number | string)}`
          );
        if (hasValue(row.analysis_severity))
          reasons.push(
            `${row.analysis_severity} severity deviation from the fair-value benchmark`
          );

        return (
          <section key={String(row.id ?? index)} className="space-y-2">
            <CompactPropertyHeader
              id={String(row.id ?? "Property")}
              index={index}
              analysisLabel="Property Valuation"
              icon={Gauge}
              fallbackTitle={
                hasValue(row.project_name)
                  ? String(row.project_name)
                  : undefined
              }
            />

            {/* 1 — the answer first: the backend's own valuation flag */}
            {flag && decision && (
              <DecisionSummary
                eyebrow="Valuation verdict"
                headline={flag}
                icon={decision.icon}
                tone={toneKey(flag)}
                tagline={decision.tagline}
                stat={
                  hasValue(row.price)
                    ? { label: "Listed price", value: formatCr(row.price) }
                    : undefined
                }
              />
            )}

            {/* 2 — why */}
            {flag && <WhyCard title={`Why ${flag}?`} reasons={reasons} />}

            {/* 3 — metrics */}
            <div className="grid grid-cols-2 gap-2 xl:grid-cols-3">
              {hasValue(row.price) && (
                <MetricCard
                  label="Listed price"
                  value={formatCr(row.price)}
                  icon={Gauge}
                />
              )}
              {hasValue(row.costpersqft) && (
                <MetricCard
                  label="Cost per sq.ft."
                  value={formatPerSqft(row.costpersqft as number | string)}
                  icon={Ruler}
                />
              )}
              {hasValue(row.analysis_severity) && (
                <PillCard label="Severity" values={[row.analysis_severity]} />
              )}
            </div>

            {/* 4 — the backend's own assessment of what this means */}
            {hasValue(row.analysis_msg) && (
              <RecommendationBar
                tone={flag ? toneKey(flag) : "positive"}
                icon={Gauge}
                title="Valuation assessment"
              >
                {row.analysis_msg}
              </RecommendationBar>
            )}

            {/* 5 — the long tail, collapsed */}
            {hasExtraFields(row, KNOWN_FIELDS) && (
              <TechnicalDetails>
                <KeyValueList record={row} omit={KNOWN_FIELDS} />
              </TechnicalDetails>
            )}
          </section>
        );
      })}

      <MetricExplainer
        items={[
          {
            term: "Valuation verdict",
            meaning:
              "Whether the listed price sits below (undervalued), within (fair) or above (overpriced) the fair-value benchmark computed by EstateMind's analysis engine.",
          },
          {
            term: "Severity",
            meaning:
              "How far the price deviates from the fair range — a high severity means a large gap and deserves closer inspection either way.",
          },
          {
            term: "How to use it",
            meaning:
              "An undervalued flag can indicate an opportunity, while an overpriced flag is a starting point for negotiation rather than an automatic pass.",
          },
        ]}
      />
    </div>
  );
}
