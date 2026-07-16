"use client";

// Property Comparison workspace (Phase 17.0) — the dedicated /compare page.
// Professional side-by-side comparison of 2–3 staged properties: property
// selector → Executive Winner Summary → comparison matrix → AI analysis
// comparison sections → Final Scoreboard → Final Recommendation. 100% frontend
// presentation and orchestration: the backend comparison, analysis rows and
// property records are fetched from the EXISTING documented endpoints
// (use-compare-data) and rendered unchanged — no analysis logic lives here.

import { useRef, useState } from "react";
import { PackageOpen, Scale, Sparkles } from "lucide-react";

import { Checkbox } from "@/components/ui/checkbox";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SectionLabel } from "@/features/analysis/ui/analysis-ui";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { AnalysisComparisonSections } from "./analysis-sections";
import { CompareExport } from "./compare-export";
import { CompareSelector } from "./compare-selector";
import {
  ExecutiveWinner,
  FinalRecommendation,
  FinalScoreboard,
} from "./compare-summary";
import { hasText } from "./compare-utils";
import { FloatingWinner } from "./floating-winner";
import type { MatrixColumn } from "./matrix-table";
import { OverallScoreboard } from "./overall-scoreboard";
import { PropertyMatrix } from "./property-matrix";
import { useCompareData } from "./use-compare-data";
import { WinnerStrip } from "./winner-strip";

// Loading placeholder shaped like the page: summary banner + two matrices.
function CompareSkeleton() {
  return (
    <div className="space-y-3" aria-busy="true" aria-label="Loading comparison">
      <Skeleton className="h-40 rounded-xl" />
      <Skeleton className="h-72 rounded-xl" />
      <Skeleton className="h-56 rounded-xl" />
    </div>
  );
}

// Persisted "Show only differences" preference (Phase 17.2, Goal 3).
const ONLY_DIFFERENCES_KEY = "estatemind.compare.only-differences";

function loadOnlyDifferences(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(ONLY_DIFFERENCES_KEY) === "true";
  } catch {
    return false;
  }
}

export function CompareWorkspace() {
  const { tray, properties } = useWorkspace();
  const [chosen, setChosen] = useState<string[]>([]);
  const [onlyDifferences, setOnlyDifferences] = useState(loadOnlyDifferences);
  const compare = useCompareData();
  // Anchor for the floating winner card: it appears once the Executive
  // Summary has scrolled out of view.
  const summaryRef = useRef<HTMLDivElement | null>(null);

  // Drop selections that were removed from the tray since they were picked.
  const selection = chosen.filter((id) => tray.includes(id));
  const bundle = compare.data;

  const runComparison = () => {
    if (selection.length >= 2) compare.mutate(selection);
  };
  const retry = compare.variables
    ? () => compare.mutate(compare.variables)
    : undefined;

  const toggleOnlyDifferences = (checked: boolean) => {
    setOnlyDifferences(checked);
    try {
      window.localStorage.setItem(ONLY_DIFFERENCES_KEY, String(checked));
    } catch {
      // Storage unavailable — the toggle still works for this session.
    }
  };

  // Remove one property from the running comparison (header quick action).
  // With 2+ properties left the comparison re-runs on the remaining ids;
  // otherwise the result clears and the user picks new contenders.
  const removeFromComparison = (id: string) => {
    const remaining = selection.filter((current) => current !== id);
    setChosen(remaining);
    if (remaining.length >= 2) compare.mutate(remaining);
    else compare.reset();
  };

  // Resolve a property id to its best available backend name: the fetched
  // property record first, the workspace search row next, the raw id last.
  const resolveName = (id: string): string => {
    const index = bundle?.ids.indexOf(id) ?? -1;
    const fromDetail = index >= 0 ? bundle?.details[index]?.project_name : null;
    if (hasText(fromDetail)) return String(fromDetail);
    return properties.find((p) => p.id === id)?.project_name || id;
  };

  // One column per compared property, shared by every comparison table. The
  // overall backend comparison winner is flagged so headers and columns can
  // badge and tint it; the advisor verdict feeds the premium header card.
  const columns: MatrixColumn[] =
    bundle?.ids.map((id, i) => {
      const detail = bundle.details[i];
      const fallback = properties.find((p) => p.id === id);
      return {
        id,
        name: resolveName(id),
        sub: hasText(detail?.location)
          ? String(detail?.location)
          : fallback?.location,
        imageUrl: detail?.image_urls?.[0] ?? fallback?.image_urls?.[0],
        isWinner: id === bundle.comparison.winner.id,
        verdict: bundle.advisor.find((row) => row.id === id)?.verdict,
      };
    }) ?? [];

  return (
    <TooltipProvider>
      <div className="space-y-4">
        <header className="card-accent-top rounded-xl border bg-card p-5 shadow-float">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            Property Comparison
          </h1>
          <p className="text-sm text-muted-foreground">
            Compare multiple shortlisted properties side-by-side before making
            an investment decision.
          </p>
        </header>

        {tray.length === 0 ? (
          <div className="rounded-xl border bg-card p-8 shadow-sm">
            <EmptyState
              icon={PackageOpen}
              title="Your tray is empty"
              description="Stage properties from the Dashboard search results to compare them here."
            />
          </div>
        ) : (
          <>
            <CompareSelector
              chosen={selection}
              onChange={setChosen}
              onShow={runComparison}
              isComparing={compare.isPending}
            />

            {compare.isPending && (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Comparing {compare.variables?.length ?? selection.length}{" "}
                  properties — running the backend analyses…
                </p>
                <CompareSkeleton />
              </div>
            )}

            {compare.isError && !compare.isPending && (
              <ErrorState
                title="Comparison failed"
                description={
                  compare.error instanceof Error
                    ? compare.error.message
                    : "Something went wrong while comparing your properties. Please try again."
                }
                onRetry={retry}
                retrying={compare.isPending}
              />
            )}

            {bundle && !compare.isPending && (
              // Re-keying on the compared ids re-runs the entrance animation
              // when the user switches the compared properties.
              <div
                key={bundle.ids.join("|")}
                className="space-y-5 animate-in fade-in slide-in-from-bottom-2 duration-300"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <SectionLabel>Comparison Result</SectionLabel>
                  <CompareExport ids={bundle.ids} />
                </div>

                <div ref={summaryRef}>
                  <ExecutiveWinner bundle={bundle} resolveName={resolveName} />
                </div>

                <WinnerStrip bundle={bundle} resolveName={resolveName} />

                <OverallScoreboard bundle={bundle} resolveName={resolveName} />

                <div className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <SectionLabel>Comparison Breakdown</SectionLabel>
                    <Label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted-foreground">
                      <Checkbox
                        checked={onlyDifferences}
                        onCheckedChange={(checked) =>
                          toggleOnlyDifferences(checked === true)
                        }
                      />
                      Show only differences
                    </Label>
                  </div>
                  <AnalysisComparisonSections
                    bundle={bundle}
                    columns={columns}
                    resolveName={resolveName}
                    hideIdentical={onlyDifferences}
                    overview={
                      <PropertyMatrix
                        columns={columns}
                        sources={bundle.ids.map((id, i) => ({
                          detail: bundle.details[i],
                          fallback: properties.find((p) => p.id === id),
                        }))}
                        hideIdentical={onlyDifferences}
                        onRemove={removeFromComparison}
                      />
                    }
                  />
                </div>

                <FinalScoreboard bundle={bundle} resolveName={resolveName} />
                <FinalRecommendation
                  bundle={bundle}
                  resolveName={resolveName}
                />

                <FloatingWinner
                  bundle={bundle}
                  resolveName={resolveName}
                  watchRef={summaryRef}
                />
              </div>
            )}

            {!bundle && !compare.isPending && !compare.isError && (
              <EmptyState
                icon={compare.isIdle ? Scale : Sparkles}
                title="Pick your contenders"
                description="Select 2–3 staged properties above, then show the comparison to see the full side-by-side breakdown."
                className="py-12"
              />
            )}
          </>
        )}
      </div>
    </TooltipProvider>
  );
}
