// AI Analysis API calls. Thin wrappers over the EstateMind Copilot API's
// per-property analysis endpoints (APIRouter prefix "/analysis" in
// src/api/analysis_api.py; see project_docs/03_API.md). Each just sends the
// selected property ids; the backend owns all analysis logic (the MCP tools
// get_price_prediction, get_rental_analysis, get_valuation_analysis,
// get_negotiation_strategy and get_investment_advice — see
// src/mcp/tools/property_tools.py). The frontend only renders the returned rows.
//
// Phase 15.4 — each request opts in to an AI explanation (`explain=true`). The
// backend returns the same analysis rows under `content` plus an optional
// `ai_explanation` string (Claude describes what the backend analysis means).
// Claude is optional: `ai_explanation` is null when it is unavailable, and the
// analysis rows are always returned, so an AI failure never blocks the analysis.

import { apiRequest } from "@/lib/api-client";
import type { AdvisorRow, AnalysisRow, NegotiationRow } from "@/types/dashboard";

// Backend analysis response when an explanation is requested. `content` is the
// unchanged backend analysis; `ai_explanation` is Claude's optional summary.
export type AnalysisResponse<T> = {
  content: T;
  ai_explanation?: string | null;
};

// Shared caller: requests the analysis with an AI explanation. `analysisType`
// only labels the explanation (e.g. risk vs. advisor share the /advisor
// endpoint); it never changes the backend analysis.
function analyzeWithExplanation<T>(
  path: string,
  ids: string[],
  analysisType: string
): Promise<AnalysisResponse<T>> {
  const query = new URLSearchParams({
    explain: "true",
    analysis_type: analysisType,
  });

  return apiRequest<AnalysisResponse<T>>(`${path}?${query.toString()}`, {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /predict — predicted vs. original price per property.
export function predictProperties(
  ids: string[]
): Promise<AnalysisResponse<AnalysisRow[]>> {
  return analyzeWithExplanation("/analysis/predict", ids, "prediction");
}

// POST /rental — rental estimate, yield, ROI, investment score per property.
export function analyzeRental(
  ids: string[]
): Promise<AnalysisResponse<AnalysisRow[]>> {
  return analyzeWithExplanation("/analysis/rental", ids, "rental");
}

// POST /valuation — fair value / benchmark-deviation assessment per property.
export function analyzeValuation(
  ids: string[]
): Promise<AnalysisResponse<AnalysisRow[]>> {
  return analyzeWithExplanation("/analysis/valuation", ids, "valuation");
}

// POST /advisor — investment recommendation (suitability, verdict, positives,
// risks) per property. Risk Analysis and Future Growth are surfaced from these
// same rows: the backend embeds risk metrics AND the future-growth fields
// (growth_label / growth_reason, produced by run_future_agent during
// enrichment) inside the investment advice (get_investment_advice).
// `analysisType` selects a risk-, growth- or advisor-focused explanation of the
// identical backend rows ("future" maps to the backend's Future Growth label).
export function getInvestmentAdvice(
  ids: string[],
  analysisType: "advisor" | "risk" | "future" = "advisor"
): Promise<AnalysisResponse<AdvisorRow[]>> {
  return analyzeWithExplanation("/analysis/advisor", ids, analysisType);
}

// POST /advisor?summary=true — AI Investment Advisor summary (Phase 15.6). The
// backend returns the same Investment Advisor rows under `content`, and
// combines the other existing analyses (price prediction, valuation, rental,
// negotiation) so Claude can write ONE investment summary over them under
// `ai_explanation`. The advisor rows are unchanged; Claude only summarizes and
// never predicts, values, scores or overrides the backend recommendation.
// Claude is optional: `ai_explanation` is null when it is unavailable.
export function getInvestmentSummary(
  ids: string[]
): Promise<AnalysisResponse<AdvisorRow[]>> {
  const query = new URLSearchParams({ summary: "true" });

  return apiRequest<AnalysisResponse<AdvisorRow[]>>(
    `/analysis/advisor?${query.toString()}`,
    {
      method: "POST",
      body: { property_ids: ids },
    }
  );
}

// POST /negotiation — target price, discount, leverage and talking points.
export function getNegotiationStrategy(
  ids: string[]
): Promise<AnalysisResponse<NegotiationRow[]>> {
  return analyzeWithExplanation("/analysis/negotiation", ids, "negotiation");
}
