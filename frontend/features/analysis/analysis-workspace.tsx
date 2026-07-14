"use client";

import { useEffect, useState } from "react";

// AI Analysis page body (Phase 8). Reuses the shared evaluation tray (staged
// from AI Chat search results) to pick which properties to analyze, then runs
// the documented per-property analysis endpoints and renders each result with
// the existing reusable renderers (AnalysisTable / AdvisorCards /
// NegotiationCards). No analysis logic lives here — the backend owns it; this
// only triggers a request and displays the response, mirroring the Streamlit
// copilot where each analysis is a single tool call against the staged tray.
//
// The former Property Comparison page now lives here as the FIRST card: it
// reuses the existing useComparison hook (POST /analysis/comparison via
// compare-service) and the extracted ComparisonView renderer, so the whole
// analysis workflow — compare first, then per-property analyses — happens in
// this single workspace against the same tray selection.

import {
  Brain,
  Gauge,
  Handshake,
  KeyRound,
  LineChart,
  Scale,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Spinner } from "@/components/ui/spinner";
import { EvaluationTray } from "@/features/dashboard/evaluation-tray";
import { AnalysisTable } from "@/features/dashboard/analysis-table";
import { AdvisorCards, NegotiationCards } from "@/features/dashboard/analysis-cards";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import type {
  AdvisorRow,
  AnalysisRow,
  NegotiationRow,
} from "@/types/dashboard";
import { cn } from "@/lib/utils";
import { ComparisonView } from "@/features/comparison/comparison-view";
import { useComparison } from "@/features/comparison/use-comparison";
import { AnalysisExplanation } from "./analysis-explanation";
import { RiskCards } from "./risk-cards";
import { FutureGrowthCards } from "./future-growth-cards";
import { useAnalysis, type AnalysisKey, type AnalysisRows } from "./use-analysis";

// Cards offered by the workspace: the comparison card plus the per-property
// analyses. "compare" is not an AnalysisKey — it runs through the existing
// comparison mutation, not the analysis mutation.
type CardKey = "compare" | AnalysisKey;

type AnalysisMeta = {
  key: CardKey;
  label: string;
  icon: LucideIcon;
  blurb: string;
};

// Property Comparison, relocated from the removed /compare page (the backend
// comparison needs at least two selected properties).
const COMPARE_CARD: AnalysisMeta = {
  key: "compare",
  label: "Compare Properties",
  icon: Scale,
  blurb: "Score, rank and pick the winner among the selected properties.",
};

// The seven Phase 8 analyses, in the order requested. "growth" (Future Growth)
// reuses the advisor endpoint: the backend embeds the growth fields produced by
// run_future_agent inside the investment advice rows (Phase 15.15).
const ANALYSES: AnalysisMeta[] = [
  {
    key: "prediction",
    label: "Price Prediction",
    icon: TrendingUp,
    blurb: "Predicted price vs. the original asking price and the deal margin.",
  },
  {
    key: "rental",
    label: "Rental Analysis",
    icon: KeyRound,
    blurb: "Rental estimate, yield, ROI and overall investment score.",
  },
  {
    key: "valuation",
    label: "Property Valuation",
    icon: Gauge,
    blurb: "Fair-value benchmark deviation and pricing assessment.",
  },
  {
    key: "risk",
    label: "Risk Analysis",
    icon: ShieldAlert,
    blurb: "Key investment risks, surfaced from the advisor engine.",
  },
  {
    key: "growth",
    label: "Future Growth",
    icon: LineChart,
    blurb: "Long-term appreciation outlook and infrastructure signals.",
  },
  {
    key: "advisor",
    label: "Investment Advisor",
    icon: Brain,
    blurb: "Suitability, verdict, positives and risks per property.",
  },
  {
    key: "negotiation",
    label: "Negotiation Strategy",
    icon: Handshake,
    blurb: "Target price, leverage and talking points for the deal.",
  },
];

// Render the result payload with the renderer matching the active analysis.
function AnalysisResultView({
  active,
  data,
}: {
  active: AnalysisKey;
  data: AnalysisRows;
}) {
  if (Array.isArray(data) && data.length === 0) {
    return (
      <EmptyState
        icon={Sparkles}
        title="No analysis to show"
        description="The backend returned no analysis rows for the selected properties."
        className="py-8"
      />
    );
  }

  switch (active) {
    case "prediction":
    case "rental":
    case "valuation":
      return <AnalysisTable rows={data as AnalysisRow[]} />;
    case "risk":
      return <RiskCards rows={data as AdvisorRow[]} />;
    case "growth":
      return <FutureGrowthCards rows={data as AdvisorRow[]} />;
    case "advisor":
      return <AdvisorCards rows={data as AdvisorRow[]} />;
    case "negotiation":
      return <NegotiationCards rows={data as NegotiationRow[]} />;
  }
}

export function AnalysisWorkspace() {
  const { tray, selected } = useWorkspace();
  const { active, setActive, run, mutation } = useAnalysis();
  const comparison = useComparison();

  // Whether the result panel shows the comparison (Compare Properties card)
  // or a per-property analysis. The last clicked card wins.
  const [showCompare, setShowCompare] = useState(false);

  // Analyze the ticked properties when there is a selection, otherwise the whole
  // tray (matches the Streamlit tools, which run on every staged property).
  const targetIds = selected.length > 0 ? selected : tray;
  const canRun = targetIds.length > 0;
  // The backend comparison requires at least two properties.
  const canCompare = targetIds.length >= 2;

  const activeMeta = ANALYSES.find((a) => a.key === active);

  // Debug logging: run outside the JSX so React executes it instead of
  // rendering the statements as text.
  useEffect(() => {
    if (!mutation.data) return;

    console.log("ACTIVE ANALYSIS:", active);
    console.log("BACKEND RESPONSE:", mutation.data.content);
  }, [active, mutation.data]);

  // Retry re-runs the last analysis request with the same key and property ids.
  const lastRun = mutation.variables;
  const retryAnalysis = lastRun ? () => mutation.mutate(lastRun) : undefined;

  // Comparison state (reused hook; selections are never cleared by running it).
  const compareResult = comparison.data?.content;
  const lastCompareIds = comparison.variables;
  const retryComparison = lastCompareIds
    ? () => comparison.mutate(lastCompareIds)
    : undefined;

  const runComparison = (ids: string[]) => {
    setShowCompare(true);
    comparison.mutate(ids);
  };

  const handlePick = (meta: AnalysisMeta) => {
    const key = meta.key;
    if (key === "compare") {
      if (!canCompare) return;
      runComparison(targetIds);
      return;
    }
    setShowCompare(false);
    setActive(key);
    if (!canRun) return;
    run(key, targetIds);
  };

  return (
    <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="border-b p-4">
          <h1 className="font-heading text-lg font-semibold">AI Analysis</h1>
          <p className="text-sm text-muted-foreground">
            Stage properties in the tray, then compare and analyze them here.
          </p>
        </div>

        {/* Card picker: Compare Properties first, then the analyses */}
        <div className="grid gap-2 border-b p-4 sm:grid-cols-2 xl:grid-cols-3">
          {[COMPARE_CARD, ...ANALYSES].map((meta) => {
            const isCompare = meta.key === "compare";
            const isActive = isCompare
              ? showCompare
              : !showCompare && active === meta.key;
            const disabled = isCompare ? !canCompare : !canRun;
            return (
              <Button
                key={meta.key}
                variant={isActive ? "default" : "outline"}
                className={cn(
                  "h-auto flex-col items-start gap-1 whitespace-normal p-3 text-left",
                  isActive && "ring-2 ring-primary/30"
                )}
                disabled={disabled}
                onClick={() => handlePick(meta)}
              >
                <span className="flex items-center gap-2 font-medium">
                  <meta.icon className="size-4 shrink-0" />
                  {meta.label}
                </span>
                <span
                  className={cn(
                    "text-xs font-normal",
                    isActive
                      ? "text-primary-foreground/80"
                      : "text-muted-foreground"
                  )}
                >
                  {meta.blurb}
                </span>
              </Button>
            );
          })}
        </div>

        {/* Result panel */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {canRun ? (
            <p className="text-xs text-muted-foreground">
              Analyzing {targetIds.length}{" "}
              {targetIds.length === 1 ? "property" : "properties"}
              {selected.length > 0 ? " (selected)" : " (whole tray)"}. Pick an
              analysis above; click it again to re-run.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Your tray is empty. Stage properties from AI Chat search results to
              analyze them here.
            </p>
          )}

          {/* Comparison result (Compare Properties card) — reuses the
              existing comparison mutation and the extracted ComparisonView. */}
          {showCompare && comparison.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-4" />
              <span>Comparing selected properties…</span>
            </div>
          )}

          {showCompare && comparison.isError && (
            <ErrorState
              title="Comparison failed"
              description="Something went wrong while comparing your properties. Please try again."
              onRetry={retryComparison}
              retrying={comparison.isPending}
            />
          )}

          {showCompare && compareResult && !comparison.isPending && (
            <div className="space-y-3">
              <h2 className="text-sm font-semibold text-muted-foreground">
                {COMPARE_CARD.label}
              </h2>
              <ComparisonView
                content={compareResult}
                explanation={comparison.data?.ai_explanation}
              />
            </div>
          )}

          {!showCompare && mutation.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-4" />
              <span>Running {activeMeta?.label ?? "analysis"}…</span>
            </div>
          )}

          {!showCompare && mutation.isError && (
            <ErrorState
              title={`${activeMeta?.label ?? "Analysis"} failed`}
              description="Something went wrong while running this analysis. Please try again."
              onRetry={retryAnalysis}
              retrying={mutation.isPending}
            />
          )}

          {!showCompare &&
            active &&
            mutation.isSuccess &&
            mutation.data && (
              <div className="space-y-3">
                <h2 className="text-sm font-semibold text-muted-foreground">
                  {activeMeta?.label}
                </h2>
                {/* Claude explains the backend analysis; the cards below still
                    render the unchanged backend result even if it is absent.
                    The Investment Advisor (Phase 15.6) shows a combined
                    investment summary with a tailored fallback message. */}
                <AnalysisExplanation
                  explanation={mutation.data.ai_explanation}
                  unavailableMessage={
                    active === "advisor"
                      ? "Investment analysis is available, but the AI summary is temporarily unavailable."
                      : undefined
                  }
                />

                <AnalysisResultView
                  active={active}
                  data={mutation.data.content}
                />
              </div>
            )}

          {!showCompare && !active && canRun && (
            <EmptyState
              icon={Sparkles}
              title="Pick an analysis"
              description="Start with Compare Properties or choose an analysis above to evaluate your staged properties."
              className="h-full"
            />
          )}
        </div>
      </section>

      <aside className="min-h-0 overflow-hidden rounded-xl border bg-card shadow-sm">
        {/* The tray's Compare button reuses the same comparison mutation, so
            comparing from the tray or the card is identical and selections
            are never cleared. */}
        <EvaluationTray
          onCompare={runComparison}
          isComparing={comparison.isPending}
        />
      </aside>
    </div>
  );
}
