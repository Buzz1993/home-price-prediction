"use client";

// Reports page body (Phase 9). Reuses the shared evaluation tray (staged from AI
// Chat search results) to pick which properties the report covers, generates an
// AI report via POST /report, previews and downloads it, then shares it to a
// phone number via POST /report/share. No report logic lives here — the backend
// composes and delivers the report; this only triggers the requests and renders
// the responses, mirroring the Streamlit report workflow
// (select → generate → preview → share).
//
// Phase 18.7: premium presentation polish — hero header with gradient icon,
// better spacing, refined typography, floating layout, improved empty states.

import { FileText, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Spinner } from "@/components/ui/spinner";
import { EvaluationTray } from "@/features/dashboard/evaluation-tray";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { ReportPreview } from "./report-preview";
import { ShareReportForm } from "./share-report-form";
import { useGenerateReport } from "./use-report";

export function ReportsWorkspace() {
  const { tray, selected } = useWorkspace();
  const generate = useGenerateReport();

  // Report the ticked properties when there is a selection, otherwise the whole
  // tray (matches the analysis/report tools, which run on every staged property).
  const targetIds = selected.length > 0 ? selected : tray;
  const canGenerate = targetIds.length > 0 && !generate.isPending;
  const report = generate.data;

  // Share the SAME properties the previewed report was generated from — not the
  // live tray/selection, which the user may have changed after generating. This
  // keeps the shared report consistent with the on-screen preview.
  const reportedIds = generate.variables ?? targetIds;

  // Phase 15.10: prefer Claude's more readable enhancement; fall back to the
  // unchanged backend report (with a friendly notice) when the AI enhancement
  // is unavailable, so report generation is never blocked.
  const displayReport = report ? report.ai_enhanced ?? report.content : null;
  const enhancementNotice =
    report && !report.ai_enhanced
      ? "The report has been generated successfully, but the AI enhancement is temporarily unavailable."
      : null;

  return (
    <div className="space-y-5">
      {/* Premium page hero (Phase 18.7) */}
      <header className="space-y-3">
        <div className="flex items-center gap-4">
          <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 via-accent to-transparent ring-1 ring-primary/10">
            <FileText className="size-7 text-primary" />
          </div>
          <div className="space-y-1">
            <h1 className="font-heading text-3xl font-semibold tracking-tight">
              AI Reports
            </h1>
            <p className="text-sm text-muted-foreground">
              Generate comprehensive property investment reports powered by AI
            </p>
          </div>
        </div>
      </header>

      <div className="grid gap-5 lg:h-[calc(100dvh-12rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card shadow-float">
          <div className="border-b bg-gradient-to-br from-muted/30 to-transparent p-6">
            {targetIds.length > 0 ? (
              <p className="mb-4 text-sm text-muted-foreground">
                Report will cover <span className="font-semibold text-foreground">{targetIds.length}</span>{" "}
                {targetIds.length === 1 ? "property" : "properties"}
                {selected.length > 0 ? " (selected)" : " (whole tray)"}
              </p>
            ) : (
              <p className="mb-4 text-sm text-muted-foreground">
                Your tray is empty. Stage properties from AI Chat to begin.
              </p>
            )}
            <Button
              onClick={() => generate.mutate(targetIds)}
              disabled={!canGenerate}
              size="default"
              className="gap-2"
            >
              {generate.isPending ? (
                <>
                  <Spinner /> Generating Report…
                </>
              ) : (
                <>
                  <Sparkles className="size-4" />
                  Generate AI Report
                </>
              )}
            </Button>
          </div>

          <div className="flex-1 space-y-5 overflow-y-auto p-6">
            {generate.isError && (
              <ErrorState
                title="Report generation failed"
                description="Something went wrong while generating your report. Please try again."
                onRetry={() => generate.mutate(targetIds)}
                retrying={generate.isPending}
              />
            )}

            {report && !generate.isPending && (
              <div className="space-y-5">
                <ReportPreview report={displayReport!} notice={enhancementNotice} />
                <ShareReportForm
                  propertyIds={reportedIds}
                  report={displayReport ?? undefined}
                />
              </div>
            )}

            {!report && !generate.isPending && !generate.isError && (
              <EmptyState
                icon={Sparkles}
                title="No report generated yet"
                description={
                  tray.length === 0
                    ? "Stage properties from AI Chat search results, then generate a comprehensive AI report here."
                    : "Press Generate AI Report to build a comprehensive investment analysis for your staged properties."
                }
                className="py-24"
              />
            )}
          </div>
        </section>

        <aside className="min-h-0 overflow-hidden rounded-xl border bg-card shadow-float">
          <EvaluationTray />
        </aside>
      </div>
    </div>
  );
}
