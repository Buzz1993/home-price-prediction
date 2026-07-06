"use client";

// AI Analysis page body (Phase 8). Reuses the shared evaluation tray (staged
// from AI Chat search results) to pick which properties to analyze, then runs
// the documented per-property analysis endpoints and renders each result with
// the existing reusable renderers (AnalysisTable / AdvisorCards /
// NegotiationCards). No analysis logic lives here — the backend owns it; this
// only triggers a request and displays the response, mirroring the Streamlit
// copilot where each analysis is a single tool call against the staged tray.

import {
  Brain,
  Gauge,
  Handshake,
  KeyRound,
  LineChart,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
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
import { RiskCards } from "./risk-cards";
import { useAnalysis, type AnalysisKey, type AnalysisResult } from "./use-analysis";

type AnalysisMeta = {
  key: AnalysisKey;
  label: string;
  icon: LucideIcon;
  blurb: string;
  blocked?: boolean;
};

// The seven Phase 8 analyses, in the order requested. "growth" (Future Growth)
// has no backend endpoint or tool, so it is flagged blocked and rendered as an
// unavailable state rather than calling an invented API.
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
    blurb: "Long-term appreciation outlook.",
    blocked: true,
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
  active: Exclude<AnalysisKey, "growth">;
  data: AnalysisResult;
}) {
  if (Array.isArray(data) && data.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        The backend returned no analysis rows for the selected properties.
      </p>
    );
  }

  switch (active) {
    case "prediction":
    case "rental":
    case "valuation":
      return <AnalysisTable rows={data as AnalysisRow[]} />;
    case "risk":
      return <RiskCards rows={data as AdvisorRow[]} />;
    case "advisor":
      return <AdvisorCards rows={data as AdvisorRow[]} />;
    case "negotiation":
      return <NegotiationCards rows={data as NegotiationRow[]} />;
  }
}

export function AnalysisWorkspace() {
  const { tray, selected } = useWorkspace();
  const { active, setActive, run, mutation } = useAnalysis();

  // Analyze the ticked properties when there is a selection, otherwise the whole
  // tray (matches the Streamlit tools, which run on every staged property).
  const targetIds = selected.length > 0 ? selected : tray;
  const canRun = targetIds.length > 0;

  const activeMeta = ANALYSES.find((a) => a.key === active);

  // Retry re-runs the last analysis request with the same key and property ids.
  const lastRun = mutation.variables;
  const retryAnalysis = lastRun ? () => mutation.mutate(lastRun) : undefined;

  const handlePick = (meta: AnalysisMeta) => {
    setActive(meta.key);
    if (meta.blocked || !canRun) return;
    run(meta.key as Exclude<AnalysisKey, "growth">, targetIds);
  };

  return (
    <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card">
        <div className="border-b p-4">
          <h1 className="font-heading text-lg font-semibold">AI Analysis</h1>
          <p className="text-sm text-muted-foreground">
            Stage properties in the tray, then run an AI analysis on them.
          </p>
        </div>

        {/* Analysis picker */}
        <div className="grid gap-2 border-b p-4 sm:grid-cols-2 xl:grid-cols-3">
          {ANALYSES.map((meta) => {
            const isActive = active === meta.key;
            return (
              <Button
                key={meta.key}
                variant={isActive ? "default" : "outline"}
                className={cn(
                  "h-auto flex-col items-start gap-1 whitespace-normal p-3 text-left",
                  isActive && "ring-2 ring-primary/30"
                )}
                disabled={!meta.blocked && !canRun}
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

          {/* Future Growth: no backend endpoint or tool exists for it. */}
          {activeMeta?.blocked && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span>
                <strong>Future Growth analysis is not available yet.</strong> The
                backend exposes no future-growth endpoint or tool, so there is no
                data to display. This view will light up once the backend adds it.
              </span>
            </div>
          )}

          {!activeMeta?.blocked && mutation.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-4" />
              <span>Running {activeMeta?.label ?? "analysis"}…</span>
            </div>
          )}

          {!activeMeta?.blocked && mutation.isError && (
            <ErrorState
              title={`${activeMeta?.label ?? "Analysis"} failed`}
              description="Something went wrong while running this analysis. Please try again."
              onRetry={retryAnalysis}
              retrying={mutation.isPending}
            />
          )}

          {!activeMeta?.blocked &&
            active &&
            active !== "growth" &&
            mutation.isSuccess &&
            mutation.data && (
              <div className="space-y-2">
                <h2 className="text-sm font-semibold text-muted-foreground">
                  {activeMeta?.label}
                </h2>
                <AnalysisResultView active={active} data={mutation.data} />
              </div>
            )}

          {!active && canRun && (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
              <Sparkles className="size-8 text-muted-foreground" />
              <p className="max-w-sm text-sm text-muted-foreground">
                Choose an analysis above to evaluate your staged properties.
              </p>
            </div>
          )}
        </div>
      </section>

      <aside className="min-h-0 overflow-hidden rounded-xl border bg-card">
        <EvaluationTray />
      </aside>
    </div>
  );
}
