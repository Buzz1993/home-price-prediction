"use client";

// Decision-first Price Prediction view (Phase 15.20). Renders the unchanged
// backend prediction rows (run_mcp_prediction: id, location, original_price,
// predicted_price, margin_diff, status on failure) in the executive-report
// hierarchy: Decision Summary (overpriced / undervalued, read from the SIGN of
// the backend margin_diff — the same derivation the former difference badge
// used) → Why checklist built from the two backend prices → metric grid →
// recommended action → collapsed technical details. Presentational only:
// every number shown is a backend value; the wording only restates the sign
// and size of margin_diff.

import {
  AlertTriangle,
  BadgeCheck,
  BadgeIndianRupee,
  Scale,
  Sparkles,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

import type { AnalysisRow } from "@/types/dashboard";
import type { ToneKey } from "@/lib/value-tone";
import { formatCr, formatPriceLabel } from "@/features/dashboard/format";
import { CompactPropertyHeader } from "./property-header";
import {
  KeyValueList,
  MetricCard,
  MetricExplainer,
  MiniCompareBars,
  SectionLabel,
  StatusPill,
  hasExtraFields,
  hasValue,
} from "./ui/analysis-ui";
import {
  DecisionSummary,
  RecommendationBar,
  TechnicalDetails,
  WhyCard,
} from "./ui/decision-summary";
import { ExecutiveSummary, usePropertyName } from "./ui/executive-summary";

// Fields given a dedicated premium treatment; everything else falls through to
// the collapsed Technical details list.
const KNOWN_FIELDS = [
  "id",
  "location",
  "original_price",
  "predicted_price",
  "margin_diff",
  "status",
];

function toNumber(value: unknown): number | null {
  const n = typeof value === "string" ? Number(value) : (value as number);
  return typeof n === "number" && !Number.isNaN(n) ? n : null;
}

// The decision copy per margin sign. margin_diff = predicted - asking, so a
// POSITIVE margin means the asking price sits BELOW the model's estimate
// (undervalued) and a negative margin means it sits above (overpriced) — the
// same reading the former difference badge colored green/red.
function readMargin(margin: number): {
  headline: string;
  tone: ToneKey;
  icon: LucideIcon;
  tagline: string;
  action: string;
} {
  if (margin > 0)
    return {
      headline: "Undervalued",
      tone: "positive",
      icon: BadgeCheck,
      tagline: "Strong buying opportunity",
      action:
        "Proceed with confidence — the asking price sits below the model's estimated market value.",
    };
  if (margin < 0)
    return {
      headline: "Overpriced",
      tone: "negative",
      icon: AlertTriangle,
      tagline: "Negotiate before buying",
      action:
        "Avoid paying the full asking price — negotiate toward the predicted market value.",
    };
  return {
    headline: "Fairly priced",
    tone: "warning",
    icon: Scale,
    tagline: "Priced at the model estimate",
    action:
      "The asking price matches the model's estimate — pay close to asking.",
  };
}

export function PredictionResults({ rows }: { rows: AnalysisRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend margin_diff is largest — i.e. the asking price
  // sitting furthest BELOW the model's estimate (or closest to it when every
  // property is overpriced). Comparing the existing backend margins is
  // presentation only; nothing new is computed.
  const contenders = rows
    .map((row, i) => ({
      row,
      id: String(row.id ?? i),
      margin: toNumber(row.margin_diff),
    }))
    .filter((c) => c.margin !== null);
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) => (c.margin! > best.margin! ? c : best))
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Better value"
            id={winner.id}
            name={resolveName(winner.id)}
            badge={readMargin(winner.margin!).headline}
            statement={
              winner.margin! > 0
                ? `Asking price ${formatPriceLabel(winner.margin!)} below the predicted market value`
                : winner.margin! < 0
                  ? `Closest to its predicted value — ${formatPriceLabel(Math.abs(winner.margin!))} above prediction`
                  : "Asking price matches the predicted market value"
            }
            stat={
              toNumber(winner.row.predicted_price) !== null
                ? {
                    label: "Predicted value",
                    value: formatCr(toNumber(winner.row.predicted_price)),
                    sub: "ML model estimate",
                  }
                : undefined
            }
            contenders={contenders.map((c) => ({
              id: c.id,
              name: resolveName(c.id),
              status: readMargin(c.margin!).headline,
              display:
                toNumber(c.row.original_price) !== null
                  ? formatCr(toNumber(c.row.original_price))
                  : undefined,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((row, index) => {
        const original = toNumber(row.original_price);
        const predicted = toNumber(row.predicted_price);
        const margin = toNumber(row.margin_diff);
        const decision = margin !== null ? readMargin(margin) : null;
        // "₹17 Lakh above estimated value" — absolute backend difference.
        const diffLabel =
          margin !== null && margin !== 0
            ? `${formatPriceLabel(Math.abs(margin))} ${
                margin < 0 ? "above" : "below"
              } estimated value`
            : undefined;

        // Why checklist — restates the two backend prices and their gap.
        const reasons: string[] = [];
        if (decision && original !== null && predicted !== null) {
          reasons.push(
            margin !== null && margin < 0
              ? `Asking price ${formatCr(original)} exceeds the predicted ${formatCr(predicted)}`
              : margin !== null && margin > 0
                ? `Predicted price ${formatCr(predicted)} exceeds the asking ${formatCr(original)}`
                : `Asking price ${formatCr(original)} matches the predicted ${formatCr(predicted)}`
          );
        }
        if (diffLabel) reasons.push(`${diffLabel} (ML model benchmark)`);
        if (hasValue(row.location))
          reasons.push(`Benchmarked against ${row.location} market data`);

        return (
          <section key={String(row.id ?? index)} className="space-y-2">
            <CompactPropertyHeader
              id={String(row.id ?? "Property")}
              index={index}
              analysisLabel="Price Prediction"
              icon={TrendingUp}
              fallbackSubtitle={
                hasValue(row.location) ? String(row.location) : undefined
              }
            />

            {/* 1 — the answer first */}
            {decision && (
              <DecisionSummary
                eyebrow="Price verdict"
                headline={decision.headline}
                icon={decision.icon}
                tone={decision.tone}
                tagline={decision.tagline}
                detail={diffLabel}
                stat={
                  predicted !== null
                    ? {
                        label: "Predicted market value",
                        value: formatCr(predicted),
                        sub: "ML model estimate",
                      }
                    : undefined
                }
              />
            )}

            {/* 2 — why */}
            {decision && (
              <WhyCard title={`Why ${decision.headline}?`} reasons={reasons} />
            )}

            {/* 3 — metrics */}
            <div className="grid grid-cols-2 gap-2 xl:grid-cols-3">
              {original !== null && (
                <MetricCard
                  label="Asking price"
                  value={formatCr(original)}
                  icon={BadgeIndianRupee}
                />
              )}
              {predicted !== null && (
                <MetricCard
                  label="Predicted price"
                  value={formatCr(predicted)}
                  sub="ML model estimate"
                  icon={Sparkles}
                  highlight
                />
              )}
              {original !== null && predicted !== null && (
                <div className="rounded-xl border bg-card px-3 py-2.5 shadow-sm">
                  <MiniCompareBars
                    items={[
                      {
                        label: "Asking",
                        value: original,
                        display: formatCr(original),
                      },
                      {
                        label: "Predicted",
                        value: predicted,
                        display: formatCr(predicted),
                        emphasis: true,
                      },
                    ]}
                  />
                </div>
              )}
            </div>

            {/* Failed predictions carry a status instead of a price. */}
            {predicted === null && hasValue(row.status) && (
              <div className="rounded-xl border bg-card px-3 py-2.5 shadow-sm">
                <StatusPill value={row.status as string} />
              </div>
            )}

            {/* 4 — what to do */}
            {decision && (
              <RecommendationBar tone={decision.tone}>
                {decision.action}
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
            term: "Predicted price",
            meaning:
              "The price EstateMind's machine-learning model estimates for this property based on its characteristics, independent of the seller's asking price.",
          },
          {
            term: "Undervalued / Overpriced",
            meaning:
              "Read from the difference between the predicted and asking price. Undervalued means the model values the property above what the seller is asking — often a sign of a good deal. Overpriced means the asking price is above the model's estimate.",
          },
          {
            term: "How to use it",
            meaning:
              "Treat the prediction as a data-driven benchmark, not a guarantee. Combine it with the valuation, risk and rental analyses before deciding.",
          },
        ]}
      />
    </div>
  );
}
