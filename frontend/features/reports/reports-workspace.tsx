"use client";

// Reports page body (Phase 9). Reuses the shared evaluation tray (staged from AI
// Chat search results) to pick which properties the report covers, generates an
// AI report via POST /report, previews and downloads it, then shares it to a
// phone number via POST /report/share. No report logic lives here — the backend
// composes and delivers the report; this only triggers the requests and renders
// the responses, mirroring the Streamlit report workflow
// (select → generate → preview → share).

import { FileText, Sparkles, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
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
  const report = generate.data?.report;

  return (
    <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card">
        <div className="border-b p-4">
          <h1 className="font-heading text-lg font-semibold">Reports</h1>
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
            <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span>
                Report generation failed. Make sure the EstateMind Copilot API
                (src/api/main.py) is running and the /report endpoint is reachable
                at the configured base URL, then try again.
              </span>
            </div>
          )}

          {report && !generate.isPending && (
            <div className="space-y-4">
              <ReportPreview report={report} />
              <ShareReportForm report={report} />
            </div>
          )}

          {!report && !generate.isPending && !generate.isError && (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
              <Sparkles className="size-8 text-muted-foreground" />
              <p className="max-w-sm text-sm text-muted-foreground">
                {tray.length === 0
                  ? "Your tray is empty. Stage properties from AI Chat search results, then generate a report here."
                  : "Press Generate Report to build an AI report for your staged properties."}
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
