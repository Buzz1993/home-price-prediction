"use client";

// Report preview + PDF export (Phase 15.17). Renders the structured report
// text returned by the backend as a premium document (report-parser.ts +
// report-document.tsx). Export PDF (Phase 16.2) downloads the PDF from the
// application's single Chromium-based generator (POST /report/pdf), which
// prints this same document — so the exported file is identical to the
// WhatsApp PDF. If the backend renderer is unavailable, it falls back to the
// browser's print-to-PDF flow, which produces the same design. All report
// content is produced by the backend — this component only presents it. When
// the text does not match the structured report layout (e.g. the non-enhanced
// fallback report) the raw text is shown instead, so the preview never breaks.

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { FileDown, Info } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { downloadReportPdf } from "@/services/report-service";
import { ReportDocument } from "./report-document";
import { cleanReportText, parseReport } from "./report-parser";

export function ReportPreview({
  report,
  notice,
}: {
  report: string;
  notice?: string | null;
}) {
  const model = useMemo(() => parseReport(report), [report]);

  // The print copy is portaled to <body> so the app shell (sidebar, scroll
  // containers) can be hidden during printing without touching the layout.
  const [mounted, setMounted] = useState(false);
  // Hydration guard: sync `mounted` after first paint so the portal only
  // renders client-side. setMounted is intentionally the only effect action.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  const [exporting, setExporting] = useState(false);

  const handleExportPdf = async () => {
    setExporting(true);
    try {
      const blob = await downloadReportPdf(report);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "EstateMind Investment Report.pdf";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      // Renderer unavailable (backend down, Playwright not installed) — the
      // browser's print-to-PDF produces the same document.
      window.print();
    } finally {
      setExporting(false);
    }
  };

  const document_ = model ? (
    <ReportDocument model={model} />
  ) : (
    <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-foreground sm:text-sm">
      {cleanReportText(report)}
    </pre>
  );

  return (
    <div className="space-y-4">
      {/* Premium toolbar (Phase 18.7) */}
      <div className="flex items-center justify-between gap-3 rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-4">
        <div className="space-y-0.5">
          <h2 className="text-base font-semibold">Report Preview</h2>
          <p className="text-xs text-muted-foreground">
            Review your comprehensive investment analysis
          </p>
        </div>
        <Button
          variant="outline"
          size="default"
          onClick={handleExportPdf}
          disabled={exporting}
          className="shrink-0 gap-2"
        >
          {exporting ? <Spinner /> : <FileDown />}
          {exporting ? "Exporting…" : "Export PDF"}
        </Button>
      </div>

      {notice && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm text-amber-700 dark:text-amber-400">
          <Info className="mt-0.5 size-5 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      {/* Premium preview container (Phase 18.7) */}
      <div className="rounded-xl border bg-gradient-to-br from-muted/20 to-transparent p-6 shadow-float">
        {document_}
      </div>

      {mounted &&
        createPortal(
          <div className="report-print-root" aria-hidden>
            {document_}
            <div className="report-print-footer justify-between border-t px-1 pt-2 text-[10px] text-muted-foreground">
              <span>EstateMind · Property Investment Report</span>
              <span>Confidential — prepared for the client</span>
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
