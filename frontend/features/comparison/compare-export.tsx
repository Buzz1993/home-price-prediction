"use client";

// Export & share toolbar for the Property Comparison workspace (Phase 17.1
// Goal 8, extended in Phase 17.2 Goal 12; Comparison Report in Phase 17.3).
// Reuses the EXISTING report pipeline end-to-end, but through the dedicated
// COMPARISON report endpoints: POST /report/comparison composes the
// comparison-layout report for the compared properties (backend analyses +
// Claude presentation), POST /report/pdf renders it through the
// application's single Chromium PDF generator, and
// POST /report/comparison/share delivers the same PDF via the official
// WhatsApp Cloud API. Print uses the browser's native dialog and the
// full-report action simply navigates to the existing Reports page — whose
// standard report flow is untouched (the compared properties are already
// staged in the shared tray it reads). No new PDF generator, no new backend
// logic — the report is generated once per comparison and reused.
//
// Phase 17.5: the lazy report generation now runs INSIDE the share mutation,
// so the button flips to "Sending…" in the same render frame as the click
// (previously isPending stayed false while the report generated for 20-30s).
// The phone input and share toggle are disabled while sending, a grey helper
// line narrates the send, and success/failure banners restate the delivery
// number. Presentation only — the same endpoints are called in the same order.

import Link from "next/link";
import { useState } from "react";
import {
  CheckCircle2,
  FileDown,
  FileText,
  Printer,
  Send,
  X,
} from "lucide-react";
import { useMutation } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "@/lib/api-client";
import {
  downloadReportPdf,
  generateComparisonReport,
  shareComparisonReport,
} from "@/services/report-service";

export function CompareExport({ ids }: { ids: string[] }) {
  // The comparison report text, generated lazily on the first export/share and
  // reused afterwards (Phase 16.1 rule: share exactly what was generated).
  const [report, setReport] = useState<string | null>(null);
  const [shareOpen, setShareOpen] = useState(false);
  const [phone, setPhone] = useState("");

  // Generate the backend comparison report once, preferring the
  // Claude-presented text — mirroring the choice the Reports page makes.
  const ensureReport = async (): Promise<string> => {
    if (report) return report;
    const generated = await generateComparisonReport(ids);
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

  // The report generation runs inside the mutation so share.isPending covers
  // the FULL send (generate → deliver) from the moment of the click.
  const share = useMutation({
    mutationFn: async (phoneNumber: string) => {
      // Generate first so the delivered PDF matches a later Export PDF exactly.
      const text = await ensureReport().catch(() => undefined);
      return shareComparisonReport({
        property_ids: ids,
        phone_number: phoneNumber,
        report: text,
      });
    },
  });
  const trimmed = phone.trim();
  const canSend = trimmed.length > 0 && !share.isPending;

  const handleShare = () => {
    if (!canSend) return;
    share.mutate(trimmed);
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
  // The number the report was actually sent to, not the edited input value.
  const sentPhone = share.variables ?? trimmed;

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
          disabled={share.isPending}
        >
          {shareOpen ? <X /> : <Send />}
          Share on WhatsApp
        </Button>
        <Button variant="outline" size="sm" onClick={() => window.print()}>
          <Printer />
          Print
        </Button>
        <Button variant="outline" size="sm" asChild>
          {/* The compared properties are already staged in the shared tray the
              Reports page reads — plain navigation, nothing re-implemented. */}
          <Link href="/reports">
            <FileText />
            Full Report
          </Link>
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
              disabled={share.isPending}
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

          {share.isPending && (
            <p className="mt-2 text-xs text-muted-foreground">
              Sending comparison report to WhatsApp…
            </p>
          )}

          {delivered && (
            <div className="mt-2 flex items-start gap-1.5 text-xs text-emerald-700">
              <CheckCircle2 className="mt-0.5 size-3.5 shrink-0" />
              <span>
                <span className="block font-medium">
                  Comparison report sent successfully
                </span>
                Delivered to {sentPhone}
              </span>
            </div>
          )}
          {shareFailed && (
            <div className="mt-2 space-y-1.5">
              <p className="text-xs font-medium text-red-600">
                Unable to send report
              </p>
              <p className="text-xs text-muted-foreground">
                {failureDetail ??
                  "Something went wrong while sending. Please check the number and try again."}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={handleShare}
                disabled={!canSend}
              >
                Try Again
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
