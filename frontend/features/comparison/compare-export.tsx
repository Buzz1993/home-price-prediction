"use client";

// Export & share toolbar for the Property Comparison workspace (Phase 17.1,
// Goal 8). Reuses the EXISTING report pipeline end-to-end: POST /report
// composes (and Claude-polishes) the investment report for the compared
// properties, POST /report/pdf renders it through the application's single
// Chromium PDF generator, and POST /report/share delivers the same PDF via the
// official WhatsApp Cloud API. No new PDF generator, no new backend logic —
// the report is generated once per comparison and reused for both actions.

import { useState } from "react";
import { CheckCircle2, FileDown, Send, X } from "lucide-react";
import { useMutation } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "@/lib/api-client";
import {
  downloadReportPdf,
  generateReport,
} from "@/services/report-service";
import { useShareReport } from "@/features/reports/use-report";

export function CompareExport({ ids }: { ids: string[] }) {
  // The comparison report text, generated lazily on the first export/share and
  // reused afterwards (Phase 16.1 rule: share exactly what was generated).
  const [report, setReport] = useState<string | null>(null);
  const [shareOpen, setShareOpen] = useState(false);
  const [phone, setPhone] = useState("");

  // Generate the backend report once, preferring the Claude-enhanced text —
  // the same choice the Reports page makes.
  const ensureReport = async (): Promise<string> => {
    if (report) return report;
    const generated = await generateReport(ids);
    const text = generated.ai_enhanced ?? generated.content;
    setReport(text);
    return text;
  };

  const exportPdf = useMutation({
    mutationFn: async () => {
      const text = await ensureReport();
      const blob = await downloadReportPdf(text);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "EstateMind Property Comparison.pdf";
      link.click();
      URL.revokeObjectURL(url);
    },
  });

  const share = useShareReport();
  const trimmed = phone.trim();
  const canSend = trimmed.length > 0 && !share.isPending;

  const handleShare = async () => {
    if (!canSend) return;
    // Generate first so the delivered PDF matches a later Export PDF exactly.
    const text = await ensureReport().catch(() => undefined);
    share.mutate({
      property_ids: ids,
      phone_number: trimmed,
      report: text,
    });
  };

  // Report delivery faithfully from the backend's ShareResult (same rule as
  // ShareReportForm) — not every HTTP 2xx means the message went out.
  const delivered =
    share.isSuccess &&
    (share.data?.status_code === undefined ||
      (share.data.status_code >= 200 && share.data.status_code < 300));
  const shareFailed = share.isError || (share.isSuccess && !delivered);
  const failureDetail =
    share.error instanceof ApiError ? share.error.message : undefined;

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex flex-wrap items-center justify-end gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => exportPdf.mutate()}
          disabled={exportPdf.isPending}
        >
          {exportPdf.isPending ? <Spinner /> : <FileDown />}
          {exportPdf.isPending ? "Exporting…" : "Export Comparison PDF"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShareOpen((open) => !open)}
        >
          {shareOpen ? <X /> : <Send />}
          Share on WhatsApp
        </Button>
      </div>

      {exportPdf.isError && (
        <p className="text-xs text-red-600">
          The PDF export failed. Please make sure the backend is running and
          try again.
        </p>
      )}

      {shareOpen && (
        <div className="w-full max-w-md rounded-xl border bg-card p-3 shadow-sm sm:w-auto">
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              type="tel"
              inputMode="tel"
              placeholder="e.g. +91 98765 43210"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="sm:w-56"
            />
            <Button size="sm" onClick={handleShare} disabled={!canSend}>
              {share.isPending ? (
                <>
                  <Spinner /> Sending…
                </>
              ) : (
                <>
                  <Send /> Send
                </>
              )}
            </Button>
          </div>

          {delivered && (
            <p className="mt-2 flex items-center gap-1.5 text-xs text-emerald-700">
              <CheckCircle2 className="size-3.5 shrink-0" />
              Comparison report sent to{" "}
              {share.variables?.phone_number ?? trimmed}.
            </p>
          )}
          {shareFailed && (
            <p className="mt-2 text-xs text-red-600">
              {failureDetail ??
                "Unable to send the report. Please check the number and try again."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
