// AI Analysis API calls. Thin wrappers over the documented per-property analysis
// endpoints (project_docs/03_API.md). Each just sends the selected property ids;
// the backend owns all analysis logic (the MCP tools get_price_prediction,
// get_rental_analysis, get_valuation_analysis, get_negotiation_strategy and
// get_investment_advice — see src/mcp/tools/property_tools.py, routed by
// src/services/chat_service.py). The frontend only renders the returned rows.

import { apiRequest } from "@/lib/api-client";
import type { AdvisorRow, AnalysisRow, NegotiationRow } from "@/types/dashboard";

// POST /predict — predicted vs. original price per property.
export function predictProperties(ids: string[]): Promise<AnalysisRow[]> {
  return apiRequest<AnalysisRow[]>("/predict", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /rental — rental estimate, yield, ROI, investment score per property.
export function analyzeRental(ids: string[]): Promise<AnalysisRow[]> {
  return apiRequest<AnalysisRow[]>("/rental", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /valuation — fair value / benchmark-deviation assessment per property.
export function analyzeValuation(ids: string[]): Promise<AnalysisRow[]> {
  return apiRequest<AnalysisRow[]>("/valuation", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /advisor — investment recommendation (suitability, verdict, positives,
// risks) per property. Risk Analysis is surfaced from these same rows: the
// backend embeds risk metrics inside the investment advice (get_investment_advice).
export function getInvestmentAdvice(ids: string[]): Promise<AdvisorRow[]> {
  return apiRequest<AdvisorRow[]>("/advisor", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /negotiation — target price, discount, leverage and talking points.
export function getNegotiationStrategy(ids: string[]): Promise<NegotiationRow[]> {
  return apiRequest<NegotiationRow[]>("/negotiation", {
    method: "POST",
    body: { property_ids: ids },
  });
}
