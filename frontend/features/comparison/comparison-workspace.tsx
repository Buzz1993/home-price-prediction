"use client";

// Property Comparison page body (Phase 7). Reuses the shared evaluation tray
// (staged from AI Chat search results) to pick which properties to compare, then
// runs the documented POST /compare endpoint and renders the result: Property
// Score Cards + AI Recommendation + Comparison Table (UI doc §9). No comparison
// logic lives here — the backend scores and ranks; the tray logic and the
// ComparisonResult renderer are reused from Phases 4–6.

import { Scale } from "lucide-react";

import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Spinner } from "@/components/ui/spinner";
import { EvaluationTray } from "@/features/dashboard/evaluation-tray";
import { ComparisonResult } from "@/features/dashboard/comparison-result";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { PropertyScoreCards } from "./property-score-cards";
import { useComparison } from "./use-comparison";

export function ComparisonWorkspace() {
  const { tray } = useWorkspace();
  const comparison = useComparison();
  const result = comparison.data;

  // Retry re-runs the last comparison request with the same property ids.
  const lastIds = comparison.variables;
  const retryComparison = lastIds
    ? () => comparison.mutate(lastIds)
    : undefined;

  return (
    <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="border-b p-4">
          <h1 className="font-heading text-lg font-semibold">
            Property Comparison
          </h1>
          <p className="text-sm text-muted-foreground">
            Select at least two staged properties and run an AI comparison.
          </p>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {comparison.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-4" />
              <span>Comparing selected properties…</span>
            </div>
          )}

          {comparison.isError && (
            <ErrorState
              title="Comparison failed"
              description="Something went wrong while comparing your properties. Please try again."
              onRetry={retryComparison}
              retrying={comparison.isPending}
            />
          )}

          {result && !comparison.isPending && (
            <div className="space-y-6">
              <div className="space-y-2">
                <h2 className="text-sm font-semibold text-muted-foreground">
                  Property Score Cards
                </h2>
                <PropertyScoreCards
                  rankings={result.rankings}
                  winnerId={result.winner.id}
                />
              </div>

              <div className="space-y-2">
                <h2 className="text-sm font-semibold text-muted-foreground">
                  AI Recommendation & Ranking
                </h2>
                <ComparisonResult data={result} />
              </div>
            </div>
          )}

          {!result && !comparison.isPending && !comparison.isError && (
            <EmptyState
              icon={Scale}
              title="Nothing to compare yet"
              description={
                tray.length === 0
                  ? "Your tray is empty. Stage properties from AI Chat search results, then select them here to compare."
                  : "Tick at least two staged properties, then press Compare Properties."
              }
              className="h-full"
            />
          )}
        </div>
      </section>

      <aside className="min-h-0 overflow-hidden rounded-xl border bg-card shadow-sm">
        <EvaluationTray
          onCompare={(ids) => comparison.mutate(ids)}
          isComparing={comparison.isPending}
        />
      </aside>
    </div>
  );
}
