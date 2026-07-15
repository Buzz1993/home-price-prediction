"use client";

// Data orchestration for the Property Comparison workspace (Phase 17.0). One
// mutation gathers everything the page renders — the backend comparison
// (winner + rankings), the per-property analysis rows and the full property
// records — from the EXISTING documented endpoints. No analysis logic lives
// here: the backend computes every value; this only sequences requests.
//
// Request order matters for performance, not correctness: the backend keeps a
// per-property enrichment cache that the FIRST analysis call populates, so the
// comparison runs first (warming the cache for all selected ids) and the five
// row fetches after it are cheap cache hits instead of parallel cold
// enrichments. Property details are independent of that cache and load in
// parallel from the start.

import { useMutation } from "@tanstack/react-query";

import {
  comparePropertiesRequest,
  fetchAdvisorRows,
  fetchNegotiationRows,
  fetchPredictionRows,
  fetchRentalRows,
  fetchValuationRows,
} from "@/services/compare-service";
import { getPropertyDetails } from "@/services/property-service";
import type {
  AdvisorRow,
  AnalysisRow,
  ComparisonResult,
  NegotiationRow,
  PropertyDetail,
} from "@/types/dashboard";

export type CompareBundle = {
  // The compared property ids, in the order the user selected them. Every
  // aligned array below follows this order.
  ids: string[];
  // Full backend property records (GET /property/{id}); null when a record
  // could not be loaded — the matrix falls back to the workspace search row.
  details: (PropertyDetail | null)[];
  // The unchanged backend comparison (winner + rankings) and Claude's optional
  // explanation of it.
  comparison: ComparisonResult;
  comparisonExplanation: string | null;
  // Unchanged backend analysis rows. An analysis that failed resolves to an
  // empty list so a single failing engine never blocks the whole comparison.
  prediction: AnalysisRow[];
  rental: AnalysisRow[];
  valuation: AnalysisRow[];
  advisor: AdvisorRow[];
  negotiation: NegotiationRow[];
};

async function fetchCompareBundle(ids: string[]): Promise<CompareBundle> {
  // Property records don't touch the analysis cache — start them immediately.
  const detailsPromise = Promise.all(
    ids.map((id) =>
      getPropertyDetails(id)
        .then((record) => ("error" in record ? null : record))
        .catch(() => null)
    )
  );

  // Comparison FIRST: it enriches every selected property server-side, making
  // the row fetches below cache hits. It is also the one result the page
  // cannot render without, so its failure fails the whole request.
  const comparison = await comparePropertiesRequest(ids);
  const content = comparison.content as ComparisonResult & { error?: string };
  if (!content || content.error || !content.winner) {
    throw new Error(
      content?.error ?? "The backend returned no comparison result."
    );
  }

  const [prediction, rental, valuation, advisor, negotiation, details] =
    await Promise.all([
      fetchPredictionRows(ids).catch(() => []),
      fetchRentalRows(ids).catch(() => []),
      fetchValuationRows(ids).catch(() => []),
      fetchAdvisorRows(ids).catch(() => []),
      fetchNegotiationRows(ids).catch(() => []),
      detailsPromise,
    ]);

  return {
    ids,
    details,
    comparison: content,
    comparisonExplanation: comparison.ai_explanation ?? null,
    prediction,
    rental,
    valuation,
    advisor,
    negotiation,
  };
}

export function useCompareData() {
  return useMutation({ mutationFn: fetchCompareBundle });
}
