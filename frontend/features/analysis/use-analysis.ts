"use client";

// Analysis mutation for the AI Analysis page. One mutation drives every analysis
// type; the active key selects which documented endpoint to call and how to
// render the result. All analysis logic stays in the backend — this only tracks
// the request state (matches the Streamlit copilot, where each analysis is a
// single tool call against the staged tray).

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import {
  analyzeRental,
  analyzeValuation,
  getInvestmentAdvice,
  getInvestmentSummary,
  getNegotiationStrategy,
  predictProperties,
  type AnalysisResponse,
} from "@/services/analysis-service";
import type {
  AdvisorRow,
  AnalysisRow,
  NegotiationRow,
} from "@/types/dashboard";

// The analysis types offered by the page. "risk" and "growth" (Future Growth)
// both reuse the advisor endpoint because the backend embeds risk metrics and
// the future-growth fields (growth_label / growth_reason) inside the investment
// advice; each just requests a differently focused explanation of the same rows.
export type AnalysisKey =
  | "prediction"
  | "rental"
  | "valuation"
  | "risk"
  | "growth"
  | "advisor"
  | "negotiation";

// The backend analysis rows (unchanged) rendered by the result renderers.
export type AnalysisRows = AnalysisRow[] | AdvisorRow[] | NegotiationRow[];

// The full backend response: analysis rows plus Claude's optional explanation
// (Phase 15.4). The rows are always present; `ai_explanation` may be null.
export type AnalysisResult = AnalysisResponse<AnalysisRows>;

// Runners for the analyses. "risk" reuses the /advisor endpoint but requests a
// risk-focused explanation of the same backend rows. "growth" (Future Growth,
// Phase 15.15) likewise reuses /advisor: the advisor rows already carry the
// growth_label / growth_reason produced by run_future_agent during enrichment,
// and analysis_type="future" asks Claude for a Future Growth explanation of
// them. "advisor" (Investment Advisor, Phase 15.6) requests the combined
// investment summary: the backend returns the same advisor rows and Claude
// summarizes them together with the other existing analyses.
const RUNNERS: Record<
  AnalysisKey,
  (ids: string[]) => Promise<AnalysisResult>
> = {
  prediction: predictProperties,
  rental: analyzeRental,
  valuation: analyzeValuation,
  risk: (ids) => getInvestmentAdvice(ids, "risk"),
  growth: (ids) => getInvestmentAdvice(ids, "future"),
  advisor: getInvestmentSummary,
  negotiation: getNegotiationStrategy,
};

export function useAnalysis() {
  const [active, setActive] = useState<AnalysisKey | null>(null);

  const mutation = useMutation({
    mutationFn: ({
      key,
      ids,
    }: {
      key: AnalysisKey;
      ids: string[];
    }) => RUNNERS[key](ids),
  });

  // Run an analysis on the given property ids. `active` is set first so the
  // result panel always reflects the analysis currently being requested.
  const run = (key: AnalysisKey, ids: string[]) => {
    setActive(key);
    mutation.mutate({ key, ids });
  };

  return { active, setActive, run, mutation };
}
