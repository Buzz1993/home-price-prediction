"use client";

// Reports page body (Phase 9). Reuses the shared evaluation tray (staged from AI
// Chat search results) to pick which properties the report covers, generates an
// AI report via POST /report, previews and downloads it, then shares it to a
// phone number via POST /report/share. No report logic lives here — the backend
// composes and delivers the report; this only triggers the requests and renders
// the responses, mirroring the Streamlit report workflow
// (select → generate → preview → share).

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
    <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card shadow-float">
        <div className="border-b p-5">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            Reports
          </h1>
          <p className="text-sm text-muted-foreground">
            Stage properties in the tray, generate an AI report, then preview,
            download or share it.
          </p>
        </div>

        <div className="border-b p-4">
          {targetIds.length > 0 ? (
            <p className="mb-3 text-xs text-muted-foreground">
              Report will cover {targetIds.length}{" "}
              {targetIds.length === 1 ? "property" : "properties"}
              {selected.length > 0 ? " (selected)" : " (whole tray)"}.
            </p>
          ) : (
            <p className="mb-3 text-xs text-muted-foreground">
              Your tray is empty. Stage properties from AI Chat search results to
              build a report.
            </p>
          )}
          <Button
            onClick={() => generate.mutate(targetIds)}
            disabled={!canGenerate}
          >
            {generate.isPending ? (
              <>
                <Spinner /> Generating…
              </>
            ) : (
              <>
                <FileText /> Generate Report
              </>
            )}
          </Button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {generate.isError && (
            <ErrorState
              title="Report generation failed"
              description="Something went wrong while generating your report. Please try again."
              onRetry={() => generate.mutate(targetIds)}
              retrying={generate.isPending}
            />
          )}

          {report && !generate.isPending && (
            <div className="space-y-4">
              <ReportPreview report={displayReport!} notice={enhancementNotice} />
              {/* The previewed report text travels with the share request so
                  the backend sends exactly this report as a WhatsApp PDF
                  without regenerating it (Phase 16.1). */}
              <ShareReportForm
                propertyIds={reportedIds}
                report={displayReport ?? undefined}
              />
            </div>
          )}

          {!report && !generate.isPending && !generate.isError && (
            <EmptyState
              icon={Sparkles}
              title="No report yet"
              description={
                tray.length === 0
                  ? "Your tray is empty. Stage properties from AI Chat search results, then generate a report here."
                  : "Press Generate Report to build an AI report for your staged properties."
              }
              className="h-full"
            />
          )}
        </div>
      </section>

      <aside className="min-h-0 overflow-hidden rounded-xl border bg-card shadow-float">
        <EvaluationTray />
      </aside>
    </div>
  );
}
