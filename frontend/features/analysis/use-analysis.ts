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
  getNegotiationStrategy,
  predictProperties,
  type AnalysisResponse,
} from "@/services/analysis-service";
import type {
  AdvisorRow,
  AnalysisRow,
  NegotiationRow,
} from "@/types/dashboard";

// The analysis types offered by the page. "growth" (Future Growth) has no
// backend endpoint or tool, so it carries no runner and is rendered as a
// blocked state by the workspace. "risk" reuses the advisor endpoint because
// the backend embeds risk metrics inside the investment advice.
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

// Runners for the analyses backed by a documented endpoint. "growth" is absent
// on purpose (no backend support); the workspace never calls run() for it.
// "risk" and "advisor" share the /advisor endpoint but request a different
// explanation focus for the same backend rows.
const RUNNERS: Record<
  Exclude<AnalysisKey, "growth">,
  (ids: string[]) => Promise<AnalysisResult>
> = {
  prediction: predictProperties,
  rental: analyzeRental,
  valuation: analyzeValuation,
  risk: (ids) => getInvestmentAdvice(ids, "risk"),
  advisor: (ids) => getInvestmentAdvice(ids, "advisor"),
  negotiation: getNegotiationStrategy,
};

export function useAnalysis() {
  const [active, setActive] = useState<AnalysisKey | null>(null);

  const mutation = useMutation({
    mutationFn: ({
      key,
      ids,
    }: {
      key: Exclude<AnalysisKey, "growth">;
      ids: string[];
    }) => RUNNERS[key](ids),
  });

  // Run an analysis on the given property ids. `active` is set first so the
  // result panel always reflects the analysis currently being requested.
  const run = (key: Exclude<AnalysisKey, "growth">, ids: string[]) => {
    setActive(key);
    mutation.mutate({ key, ids });
  };

  return { active, setActive, run, mutation };
}
