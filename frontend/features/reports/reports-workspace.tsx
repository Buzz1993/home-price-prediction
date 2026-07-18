"use client";

// Reports page body (Phase 9, report center in Phase 18.9). Reuses the shared
// evaluation tray (staged from AI Chat search results) to pick which properties
// the report covers, generates an AI report via POST /report, previews and
// downloads it, then shares it to a phone number via POST /report/share. No
// report logic lives here — the backend composes and delivers the report; this
// only triggers the requests and renders the responses, mirroring the Streamlit
// report workflow (select → generate → preview → share).
//
// Phase 18.9: the page is a proper report center. Every successfully generated
// report is stored in the local Report History (report-history.ts) and listed
// newest-first with Preview / Download PDF / Share WhatsApp / Delete actions.
// Preview reopens the stored report text instantly — reports are NEVER
// regenerated. The navbar's global search filters this history list.
//
// Phase 18.11 layout refinement: two-column report center. The wide left
// column holds Generate + Report Preview (sharing now lives in the preview
// toolbar); the right sidebar stacks Recent Reports above the Evaluation
// Tray. Same components, same behavior — only the arrangement changed.

import { useEffect, useState } from "react";
import { FileText, History, SearchX, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalSearch } from "@/components/providers/search-provider";
import { EvaluationTray } from "@/features/dashboard/evaluation-tray";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import {
  deleteReport,
  loadReportHistory,
  type StoredReport,
} from "./report-history";
import { ReportHistoryList } from "./report-history-list";
import { ReportPreview } from "./report-preview";
import { useGenerateReport } from "./use-report";

export function ReportsWorkspace() {
  const { tray, selected } = useWorkspace();
  const generate = useGenerateReport();
  const { query } = useGlobalSearch();

  // Local report history (newest first). Loaded after mount — localStorage is
  // client-only — and refreshed whenever a generation succeeds.
  const [history, setHistory] = useState<StoredReport[]>([]);
  // A stored report opened via Preview. Takes precedence over the freshly
  // generated preview; cleared when a new generation starts.
  const [opened, setOpened] = useState<StoredReport | null>(null);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    setHistory(loadReportHistory());
  }, [generate.isSuccess]);
  /* eslint-enable react-hooks/set-state-in-effect */

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

  const handleGenerate = () => {
    setOpened(null);
    generate.mutate(targetIds);
  };

  const handleDelete = (id: string) => {
    setHistory(deleteReport(id));
    if (opened?.id === id) setOpened(null);
  };

  // Global search filters the history list (title, type, property ids, date).
  const q = query.trim().toLowerCase();
  const filteredHistory = q
    ? history.filter((r) =>
        [
          r.title,
          r.type,
          new Date(r.createdAt).toLocaleDateString(),
          ...r.propertyIds,
        ]
          .join(" ")
          .toLowerCase()
          .includes(q)
      )
    : history;

  // What the preview panel shows: an opened stored report wins, then the
  // freshly generated one.
  const showOpened = opened && !generate.isPending;
  const showGenerated = !opened && report && !generate.isPending;

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

      {/* Two-column report center (Phase 18.11): wide report area (~72%) +
          right sidebar (~28%) stacking Recent Reports above the Evaluation
          Tray. */}
      <div className="grid gap-5 lg:h-[calc(100dvh-12rem)] lg:grid-cols-[minmax(0,72fr)_minmax(0,28fr)]">
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
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={handleGenerate}
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
              </TooltipTrigger>
              <TooltipContent className="max-w-[220px]">
                Create a comprehensive investment report.
              </TooltipContent>
            </Tooltip>
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

            {/* A stored report reopened from the history — rendered from the
                saved text, never regenerated. Sharing lives in the preview
                toolbar (Phase 18.11). */}
            {showOpened && (
              <ReportPreview
                report={opened.content}
                sharePropertyIds={opened.propertyIds}
                shareType={opened.type}
              />
            )}

            {showGenerated && (
              <ReportPreview
                report={displayReport!}
                notice={enhancementNotice}
                sharePropertyIds={reportedIds}
              />
            )}

            {!showOpened && !showGenerated && !generate.isPending && !generate.isError && (
              <EmptyState
                icon={Sparkles}
                title="No report open"
                description={
                  history.length > 0
                    ? "Generate a new AI report, or reopen a previous one from Recent Reports in the sidebar."
                    : tray.length === 0
                      ? "Stage properties from AI Chat search results, then generate a comprehensive AI report here."
                      : "Press Generate AI Report to build a comprehensive investment analysis for your staged properties."
                }
                className="py-16"
              />
            )}
          </div>
        </section>

        {/* Right sidebar — Recent Reports above the Evaluation Tray. Within
            the viewport-height grid each panel gets its own bounded height
            (flex-1) and scrolls internally (Phase 18.12): a long report list
            can no longer push the tray down, and either panel scrolls without
            moving the other. */}
        <aside className="flex min-h-0 flex-col gap-5">
          {/* Recent Reports — the local report center (Phase 18.9), moved to
              the sidebar in Phase 18.11. Same component, same actions. */}
          {(history.length > 0 || q) && (
            <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border bg-card shadow-float">
              <div className="flex shrink-0 flex-wrap items-center gap-2 border-b p-4">
                <History className="size-4 text-primary" />
                <h2 className="font-heading text-sm font-semibold">
                  Recent Reports
                </h2>
                <span className="text-xs text-muted-foreground">
                  {filteredHistory.length}{" "}
                  {filteredHistory.length === 1 ? "report" : "reports"}
                  {q ? ` matching “${query.trim()}”` : ""}
                </span>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto scroll-smooth p-3">
                {filteredHistory.length > 0 ? (
                  <ReportHistoryList
                    reports={filteredHistory}
                    activeId={opened?.id ?? null}
                    onPreview={setOpened}
                    onDelete={handleDelete}
                  />
                ) : (
                  <EmptyState
                    icon={SearchX}
                    description={`No reports match “${query.trim()}”.`}
                    className="py-10"
                  />
                )}
              </div>
            </section>
          )}

          <section className="min-h-0 flex-1 overflow-hidden rounded-xl border bg-card shadow-float">
            <EvaluationTray />
          </section>
        </aside>
      </div>
    </div>
  );
}
