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
//
// Phase 18.11: the Share Report action lives in this toolbar, beside Export
// PDF. Phase 18.12 makes it a compact floating popover anchored to the Share
// button (fade + scale in, closes on outside-click or Escape) instead of a
// full-width collapsible panel — a lighter, SaaS-style share widget. Sharing
// logic (validation, endpoint, states) is untouched; only the form's location
// and container changed.

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FileDown, Info, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { downloadReportPdf } from "@/services/report-service";
import { ReportDocument } from "./report-document";
import type { ReportType } from "./report-history";
import { cleanReportText, parseReport } from "./report-parser";
import { ShareReportForm } from "./share-report-form";

export function ReportPreview({
  report,
  notice,
  sharePropertyIds,
  shareType = "property",
}: {
  report: string;
  notice?: string | null;
  // When provided, the toolbar shows a Share Report button that expands the
  // existing WhatsApp share form (Phase 18.11) for these property ids.
  sharePropertyIds?: string[];
  shareType?: ReportType;
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
  // Floating share popover (Phase 18.12) — toggled from the toolbar and
  // anchored to the Share button. Closes on outside-click or Escape.
  const [shareOpen, setShareOpen] = useState(false);
  const shareWrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!shareOpen) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!shareWrapRef.current?.contains(event.target as Node)) {
        setShareOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setShareOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [shareOpen]);

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
      {/* Premium toolbar (Phase 18.7; Share moved here in Phase 18.11) */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-4">
        <div className="space-y-0.5">
          <h2 className="text-base font-semibold">Report Preview</h2>
          <p className="text-xs text-muted-foreground">
            Review your comprehensive investment analysis
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="default"
                onClick={handleExportPdf}
                disabled={exporting}
                className="gap-2"
              >
                {exporting ? <Spinner /> : <FileDown />}
                {exporting ? "Exporting…" : "Export PDF"}
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-[220px]">
              Download this report as a PDF.
            </TooltipContent>
          </Tooltip>
          {sharePropertyIds && (
            <div ref={shareWrapRef} className="relative">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={shareOpen ? "secondary" : "default"}
                    size="default"
                    onClick={() => setShareOpen((open) => !open)}
                    aria-expanded={shareOpen}
                    aria-haspopup="dialog"
                    className="gap-2"
                  >
                    <Send /> Share Report
                  </Button>
                </TooltipTrigger>
                <TooltipContent className="max-w-[220px]">
                  Send this report to WhatsApp.
                </TooltipContent>
              </Tooltip>

              {/* Compact floating share widget (Phase 18.12) — anchored to the
                  button, right-aligned, with a fade + scale entrance. Holds the
                  existing WhatsApp share form, unchanged. */}
              {shareOpen && (
                <div
                  role="dialog"
                  aria-label="Share report to WhatsApp"
                  className="absolute right-0 top-full z-40 mt-2 w-[min(22rem,calc(100vw-2rem))] origin-top-right animate-in fade-in zoom-in-95 duration-200"
                >
                  <ShareReportForm
                    propertyIds={sharePropertyIds}
                    report={report}
                    type={shareType}
                    compact
                  />
                </div>
              )}
            </div>
          )}
        </div>
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
