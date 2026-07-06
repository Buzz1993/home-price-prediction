// Comparison API call. Thin wrapper over the documented POST /compare endpoint —
// the backend runs the analytical comparison (scoring, ranking, verdict). The
// frontend only sends the selected property ids and renders the result. See
// project_docs/03_API.md and the Streamlit run_comparison reference
// (src/ui/intent_chat_ui.py).

import { apiRequest } from "@/lib/api-client";
import type { ComparisonResult } from "@/types/dashboard";

export function comparePropertiesRequest(
  ids: string[]
): Promise<ComparisonResult> {
  return apiRequest<ComparisonResult>("/compare", {
    method: "POST",
    body: { property_ids: ids },
  });
}
