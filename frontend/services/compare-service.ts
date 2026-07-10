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
import type { ComparisonResult } from "@/types/dashboard";

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
