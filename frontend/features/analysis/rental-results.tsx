"use client";

// Decision-first Rental Analysis view (Phase 15.20). Renders the unchanged
// backend rental rows (run_mcp_rental: id, monthly_rent_estimate, annual_rent,
// rental_yield_percent, demand_level, investment_rating, rental_strategy) in
// the executive-report hierarchy: Decision Summary (the verbatim backend
// investment_rating, restated as stars, with the verbatim yield as the key
// stat) → Why checklist built from the yield / rent / demand fields → metric
// grid → the verbatim backend rental_strategy as the recommended action →
// collapsed technical details. The rating is shown ONCE (in the summary), not
// repeated in the header pill and metric pills as before.

import { BadgeIndianRupee, CalendarRange, KeyRound } from "lucide-react";

import type { AnalysisRow } from "@/types/dashboard";
import { toneKey } from "@/lib/value-tone";
import { formatPercent } from "@/features/dashboard/format";
import { CompactPropertyHeader } from "./property-header";
import {
  KeyValueList,
  MetricCard,
  MetricExplainer,
  PillCard,
  RadialGauge,
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
  ratingStars,
} from "./ui/decision-summary";

const KNOWN_FIELDS = [
  "id",
  "monthly_rent_estimate",
  "annual_rent",
  "rental_yield_percent",
  "demand_level",
  "investment_rating",
  "rental_strategy",
];

// Rent estimates from the backend are absolute ₹ amounts (not Crores).
function formatRupees(value: unknown): string {
  const n = typeof value === "string" ? Number(value) : (value as number);
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

// The backend yield is a string like "3.20%"; the number is only used for the
// gauge arc — the verbatim string is what gets displayed.
function parseYield(value: unknown): number | null {
  if (!hasValue(value)) return null;
  const n = Number(String(value).replace("%", ""));
  return Number.isNaN(n) ? null : n;
}

export function RentalResults({ rows }: { rows: AnalysisRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend rental yield is highest. Comparing the existing
  // backend percentages is presentation only.
  const contenders = rows
    .map((row, i) => ({
      row,
      id: String(row.id ?? i),
      yieldValue: parseYield(row.rental_yield_percent),
    }))
    .filter((c) => c.yieldValue !== null);
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) =>
          c.yieldValue! > best.yieldValue! ? c : best
        )
      : null;
  const winnerRating =
    winner && hasValue(winner.row.investment_rating)
      ? String(winner.row.investment_rating)
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Better rental investment"
            id={winner.id}
            name={resolveName(winner.id)}
            badge={winnerRating ?? undefined}
            stars={winnerRating ? ratingStars(winnerRating) : undefined}
            statement={`Highest rental yield of ${formatPercent(winner.row.rental_yield_percent as string)} among the compared properties`}
            stat={{
              label: "Rental yield",
              value: formatPercent(winner.row.rental_yield_percent as string),
              sub: "Annual rent vs. price",
            }}
            reasons={[
              ...(hasValue(winner.row.annual_rent)
                ? [`Annual income ${formatRupees(winner.row.annual_rent)}`]
                : []),
              ...(hasValue(winner.row.demand_level)
                ? [`${winner.row.demand_level} tenant demand`]
                : []),
            ]}
            contenders={contenders.map((c) => ({
              id: c.id,
              name: resolveName(c.id),
              status: hasValue(c.row.investment_rating)
                ? c.row.investment_rating
                : undefined,
              display: `${formatPercent(c.row.rental_yield_percent as string)} yield`,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((row, index) => {
        const yieldValue = parseYield(row.rental_yield_percent);
        const rating = hasValue(row.investment_rating)
          ? String(row.investment_rating)
          : null;

        // Why checklist — restates the backend rental fields verbatim.
        const reasons: string[] = [];
        if (hasValue(row.rental_yield_percent))
          reasons.push(
            `Rental yield of ${formatPercent(row.rental_yield_percent as string)}`
          );
        if (hasValue(row.annual_rent))
          reasons.push(
            `Estimated annual income ${formatRupees(row.annual_rent)}`
          );
        if (hasValue(row.monthly_rent_estimate))
          reasons.push(
            `Expected monthly rent ${formatRupees(row.monthly_rent_estimate)}`
          );
        if (hasValue(row.demand_level))
          reasons.push(`${row.demand_level} tenant demand`);

        return (
          <section key={String(row.id ?? index)} className="space-y-2">
            <CompactPropertyHeader
              id={String(row.id ?? "Property")}
              index={index}
              analysisLabel="Rental Analysis"
              icon={KeyRound}
            />

            {/* 1 — the answer first: the backend's own investment rating */}
            {rating && (
              <DecisionSummary
                eyebrow="Investment rating"
                headline={rating}
                tone={toneKey(rating)}
                stars={ratingStars(rating)}
                tagline="Rental investment outlook"
                stat={
                  hasValue(row.rental_yield_percent)
                    ? {
                        label: "Expected rental yield",
                        value: formatPercent(row.rental_yield_percent as string),
                        sub: "Annual rent vs. price",
                      }
                    : undefined
                }
              />
            )}

            {/* 2 — why */}
            {rating && (
              <WhyCard
                title={`Why ${rating} rental investment?`}
                reasons={reasons}
              />
            )}

            {/* 3 — metrics */}
            <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
              {hasValue(row.monthly_rent_estimate) && (
                <MetricCard
                  label="Monthly rent"
                  value={formatRupees(row.monthly_rent_estimate)}
                  sub="Estimated"
                  icon={BadgeIndianRupee}
                />
              )}
              {hasValue(row.annual_rent) && (
                <MetricCard
                  label="Annual rent"
                  value={formatRupees(row.annual_rent)}
                  sub="Estimated"
                  icon={CalendarRange}
                />
              )}
              {/* Radial yield gauge — arc scaled against a 5% yield ceiling
                  for context; the verbatim backend percentage is displayed. */}
              {hasValue(row.rental_yield_percent) && yieldValue !== null && (
                <div className="flex items-center gap-3 rounded-xl border border-primary/40 bg-primary/5 px-3 py-2 shadow-sm">
                  <RadialGauge
                    value={yieldValue}
                    max={5}
                    display={formatPercent(row.rental_yield_percent as string)}
                  />
                  <div className="min-w-0">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Rental yield
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      Annual rent vs. price
                    </p>
                  </div>
                </div>
              )}
              {hasValue(row.demand_level) && (
                <PillCard
                  label="Tenant demand"
                  values={[`${row.demand_level} demand`]}
                />
              )}
            </div>

            {/* 4 — what to do: the backend's own rental strategy */}
            {hasValue(row.rental_strategy) && (
              <RecommendationBar
                tone={rating ? toneKey(rating) : "positive"}
                icon={KeyRound}
                title="Recommended rental strategy"
              >
                {row.rental_strategy}
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
            term: "Rental yield",
            meaning:
              "Annual rental income as a percentage of the property price. In Indian metros, ~2% is typical; 3%+ is considered strong for residential property.",
          },
          {
            term: "Demand level",
            meaning:
              "How strong tenant demand is expected to be for this property, based on its location and characteristics.",
          },
          {
            term: "Investment rating",
            meaning:
              "The backend's overall rental-investment grade, combining yield with growth and risk signals for the property. The stars restate this grade visually.",
          },
        ]}
      />
    </div>
  );
}
