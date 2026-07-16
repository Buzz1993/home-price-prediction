// Category winners for the Property Comparison workspace (Phase 17.0):
// section winners, the final scoreboard and the final recommendation all read
// from these. Pure presentation, and the SAME winner rules the Phase 15.21
// Executive Comparison summaries already apply in the analysis renderers:
// highest backend yield / margin / discount / score, lowest backend price, or
// the backend's own status wording via the shared toneRank. The overall
// Investment winner is the backend comparison winner verbatim — nothing is
// re-scored here.

import { toneRank } from "@/lib/value-tone";
import {
  formatCr,
  formatPercent,
  formatScore,
  splitList,
} from "@/features/dashboard/format";
import { ratingStars } from "@/features/analysis/ui/decision-summary";
import type {
  AdvisorRow,
  AnalysisRow,
  NegotiationRow,
} from "@/types/dashboard";
import { hasText, parseNumeric } from "./compare-utils";
import type { CompareBundle } from "./use-compare-data";

export type CategoryWinner = {
  id: string;
  // Backend-derived reasons, shown as ✓ points next to the winner.
  points: string[];
} | null;

// Generic "largest backend value wins" reducer. Returns null when fewer than
// two rows carry a comparable value — no winner is invented for a one-sided
// comparison.
function bestBy<T>(rows: T[], rank: (row: T) => number | null): T | null {
  const ranked = rows
    .map((row) => ({ row, rank: rank(row) }))
    .filter((r): r is { row: T; rank: number } => r.rank !== null);
  if (ranked.length < 2) return null;
  const best = ranked.reduce((a, b) => (b.rank > a.rank ? b : a));
  // All equal → no meaningful winner.
  if (ranked.every((r) => r.rank === best.rank)) return null;
  return best.row;
}

// Advisor rows carry extra backend fields (risk_label, risk_score,
// overall_score, …) beyond the typed columns — read them through the record.
function rec(row: AdvisorRow | AnalysisRow | NegotiationRow) {
  return row as unknown as Record<string, unknown>;
}

// Lowest asking price (property records).
export function priceWinner(bundle: CompareBundle): CategoryWinner {
  const priced = bundle.ids
    .map((id, i) => ({ id, price: parseNumeric(bundle.details[i]?.price) }))
    .filter((p): p is { id: string; price: number } => p.price !== null);
  if (priced.length < 2) return null;
  const best = priced.reduce((a, b) => (b.price < a.price ? b : a));
  if (priced.every((p) => p.price === best.price)) return null;
  return { id: best.id, points: [`Lowest price at ${formatCr(best.price)}`] };
}

// Largest backend margin below the predicted value (margin_diff = predicted −
// asking; positive = undervalued).
export function predictionWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.prediction, (row) =>
    parseNumeric(row.margin_diff)
  );
  if (!best) return null;
  const points = [
    `Best value vs. predicted price (${formatCr(best.margin_diff)})`,
  ];
  // Restate the winning gap with the backend's own numbers when present.
  if (hasText(best.original_price))
    points.push(`Asking ${formatCr(best.original_price)}`);
  if (hasText(best.predicted_price))
    points.push(`Predicted ${formatCr(best.predicted_price)}`);
  return { id: String(best.id), points };
}

// Highest backend rental yield.
export function rentalWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.rental, (row) =>
    parseNumeric(row.rental_yield_percent)
  );
  if (!best) return null;
  const points = [
    `Highest rental yield of ${formatPercent(best.rental_yield_percent as string)}`,
  ];
  if (hasText(best.annual_rent)) {
    const annual = parseNumeric(best.annual_rent);
    if (annual !== null)
      points.push(
        `₹${Math.round(annual).toLocaleString("en-IN")} annual rent`
      );
  }
  if (hasText(best.investment_rating))
    points.push(`${best.investment_rating} rental rating`);
  if (hasText(best.demand_level)) points.push(`${best.demand_level} demand`);
  return { id: String(best.id), points };
}

// Best backend risk wording (risk_label when present, verdict otherwise);
// fewer backend-listed concerns break ties — same rule as RiskCards.
export function riskWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.advisor, (row) => {
    const label = rec(row).risk_label;
    const status = hasText(label) ? String(label) : row.verdict;
    if (!hasText(status)) return null;
    const concerns = splitList(row.risks).length;
    return toneRank(String(status)) * 100 - concerns;
  });
  if (!best) return null;
  const label = rec(best).risk_label;
  const status = hasText(label) ? String(label) : String(best.verdict);
  const concerns = splitList(best.risks).length;
  return {
    id: best.id,
    points: [
      status,
      `${concerns} concern${concerns === 1 ? "" : "s"} flagged`,
    ],
  };
}

// Best backend growth wording; a higher backend growth_score breaks ties.
export function growthWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.advisor, (row) => {
    if (!hasText(row.growth_label)) return null;
    const score = parseNumeric(row.growth_score) ?? 0;
    return toneRank(String(row.growth_label)) * 100 + score;
  });
  if (!best) return null;
  const points = [String(best.growth_label)];
  if (hasText(best.growth_score))
    points.push(`Growth score ${formatScore(best.growth_score as number)}`);
  return { id: best.id, points };
}

// Best backend valuation flag wording (undervalued > fair > overpriced).
export function valuationWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.valuation, (row) =>
    hasText(row.analysis_flag) ? toneRank(String(row.analysis_flag)) : null
  );
  if (!best) return null;
  const points = [String(best.analysis_flag)];
  if (hasText(best.analysis_msg)) points.push(String(best.analysis_msg));
  return { id: String(best.id), points };
}

// Highest backend overall_score when the advisor rows carry it; otherwise the
// best verdict wording (star grade first, tone as tie-break — AdvisorCards).
export function advisorWinner(bundle: CompareBundle): CategoryWinner {
  const withScores = bundle.advisor.filter((row) =>
    hasText(rec(row).overall_score)
  );
  if (withScores.length >= 2) {
    const best = bestBy(withScores, (row) =>
      parseNumeric(rec(row).overall_score)
    );
    if (best) {
      return {
        id: best.id,
        points: [
          `Investment score ${formatScore(rec(best).overall_score as number)}`,
          ...(hasText(best.verdict) ? [String(best.verdict)] : []),
        ],
      };
    }
  }
  const best = bestBy(bundle.advisor, (row) => {
    if (!hasText(row.verdict)) return null;
    const verdict = String(row.verdict);
    return (ratingStars(verdict) ?? 0) * 10 + toneRank(verdict);
  });
  if (!best) return null;
  return { id: best.id, points: [String(best.verdict)] };
}

// Largest backend suggested discount; stronger negotiation_power wording
// breaks ties — same rule as NegotiationCards.
export function negotiationWinner(bundle: CompareBundle): CategoryWinner {
  const best = bestBy(bundle.negotiation, (row) => {
    const discount = parseNumeric(row.suggested_discount_percent);
    const power = hasText(row.negotiation_power)
      ? toneRank(String(row.negotiation_power))
      : 0;
    if (discount === null && power === 0) return null;
    return (discount ?? 0) * 100 + power;
  });
  if (!best) return null;
  const points: string[] = [];
  if (hasText(best.suggested_discount_percent))
    points.push(
      `${formatPercent(best.suggested_discount_percent as string)} suggested discount`
    );
  if (hasText(best.negotiation_power))
    points.push(`${best.negotiation_power} negotiation power`);
  if (hasText(best.target_price))
    points.push(`Target price ${formatCr(best.target_price)}`);
  return { id: best.id, points };
}

// The backend comparison winner, verbatim.
export function investmentWinner(bundle: CompareBundle): CategoryWinner {
  const winner = bundle.comparison.winner;
  if (!winner) return null;
  // The backend score can be null (NaN-scrubbed) — only cite it when present.
  const points = [
    ...(hasText(winner.overall_score)
      ? [`Overall score ${formatScore(winner.overall_score)}`]
      : []),
    ...(hasText(winner.verdict) ? [String(winner.verdict)] : []),
  ];
  return {
    id: winner.id,
    points:
      points.length > 0 ? points : splitList(winner.comparison_reason).slice(0, 2),
  };
}

// The final scoreboard rows, in display order.
export function scoreboard(bundle: CompareBundle) {
  return [
    { label: "Price Winner", winner: priceWinner(bundle) },
    { label: "Rental Winner", winner: rentalWinner(bundle) },
    { label: "Risk Winner", winner: riskWinner(bundle) },
    { label: "Growth Winner", winner: growthWinner(bundle) },
    { label: "Valuation Winner", winner: valuationWinner(bundle) },
    { label: "Negotiation Winner", winner: negotiationWinner(bundle) },
    { label: "Investment Winner", winner: investmentWinner(bundle) },
  ].filter(
    (entry): entry is { label: string; winner: NonNullable<CategoryWinner> } =>
      entry.winner !== null
  );
}
