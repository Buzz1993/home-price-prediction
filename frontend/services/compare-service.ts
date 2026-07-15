// Comparison API call. Thin wrapper over the POST /analysis/comparison endpoint
// (APIRouter prefix "/analysis" in src/api/analysis_api.py) — the backend runs the
// analytical comparison (scoring, ranking, verdict) via compare_properties. The
// frontend only sends the selected property ids and renders the result. See
// project_docs/03_API.md and the Streamlit run_comparison reference
// (src/ui/intent_chat_ui.py).
//
// Phase 15.5 — the request opts in to an AI explanation (`explain=true`). The
// backend returns the same comparison result under `content` plus an optional
// `ai_explanation` string (Claude explains why the backend ranked the
// properties as it did). Claude is optional: `ai_explanation` is null when it is
// unavailable, and the comparison is always returned, so an AI failure never
// blocks the comparison. Claude only explains the backend comparison — it never
// compares, scores or ranks the properties itself.

import { apiRequest } from "@/lib/api-client";
import type { AnalysisResponse } from "@/services/analysis-service";
import type {
  AdvisorRow,
  AnalysisRow,
  ComparisonResult,
  NegotiationRow,
} from "@/types/dashboard";

export function comparePropertiesRequest(
  ids: string[]
): Promise<AnalysisResponse<ComparisonResult>> {
  const query = new URLSearchParams({ explain: "true" });

  return apiRequest<AnalysisResponse<ComparisonResult>>(
    `/analysis/comparison?${query.toString()}`,
    {
      method: "POST",
      body: { property_ids: ids },
    }
  );
}

// ---------------------------------------------------------------------------
// Raw analysis rows for the Property Comparison workspace (Phase 17.0). Same
// documented /analysis endpoints, called WITHOUT `explain` (the backend
// default), so the backend returns the unchanged analysis rows directly — no
// AI explanation is generated for them. The comparison page renders these rows
// side-by-side in the matrix; the only Claude call it makes is the comparison
// explanation above (per the "minimize the context sent to Claude" API rule).
// ---------------------------------------------------------------------------

function analysisRows<T>(path: string, ids: string[]): Promise<T> {
  return apiRequest<T>(path, {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /analysis/predict — predicted vs. original price rows.
export function fetchPredictionRows(ids: string[]): Promise<AnalysisRow[]> {
  return analysisRows("/analysis/predict", ids);
}

// POST /analysis/rental — rent estimate, yield, demand and rating rows.
export function fetchRentalRows(ids: string[]): Promise<AnalysisRow[]> {
  return analysisRows("/analysis/rental", ids);
}

// POST /analysis/valuation — fair-value assessment rows.
export function fetchValuationRows(ids: string[]): Promise<AnalysisRow[]> {
  return analysisRows("/analysis/valuation", ids);
}

// POST /analysis/advisor — investment advice rows. The backend embeds the risk
// metrics AND the future-growth fields inside these same rows, so one call
// feeds the Risk, Future Growth and Investment Advisor comparison sections.
export function fetchAdvisorRows(ids: string[]): Promise<AdvisorRow[]> {
  return analysisRows("/analysis/advisor", ids);
}

// POST /analysis/negotiation — target price, discount and strategy rows.
export function fetchNegotiationRows(ids: string[]): Promise<NegotiationRow[]> {
  return analysisRows("/analysis/negotiation", ids);
}
