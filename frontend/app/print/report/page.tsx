"use client";

// Print-only report route (Phase 16.2). Opened by the backend's headless
// Chromium renderer (src/services/pdf_service.py) to export the report as a
// PDF. It renders the SAME approved report document as the Reports page —
// ReportDocument + report-parser + the existing .report-print-root print CSS
// (globals.css) — so the exported PDF is pixel-identical to the browser's
// Export PDF / Chrome print output. Comparison reports (Phase 17.4) render
// through ComparisonReportDocument instead, mirroring the /compare page; the
// standard report path is unchanged. No report layout or styling lives here.
//
// The renderer injects the already-generated report text as
// window.__ESTATEMIND_REPORT__ before the page loads; nothing is regenerated.
// data-report-ready signals that the document has rendered and printing can
// start. On screen this route is intentionally blank (.report-print-root is
// print-only).

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import {
  ComparisonReportDocument,
  isComparisonReport,
} from "@/features/reports/comparison-report-document";
import { ReportDocument } from "@/features/reports/report-document";
import { parseReport } from "@/features/reports/report-parser";

declare global {
  interface Window {
    __ESTATEMIND_REPORT__?: string;
  }
}

export default function PrintReportPage() {
  const [report, setReport] = useState<string | null>(null);

  // The injected report is only readable on the client, after mount.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setReport(window.__ESTATEMIND_REPORT__ ?? null);
  }, []);

  const model = useMemo(
    () => (report ? parseReport(report) : null),
    [report]
  );

  if (!report) return null;

  // Comparison reports (Phase 17.4) print with the compare-page renderer;
  // standard reports keep the unchanged ReportDocument. Same raw-text
  // fallback as the Reports page preview, so the exported PDF can never
  // diverge from what the user previewed.
  const comparison = model !== null && isComparisonReport(model);
  const document_ = model ? (
    comparison ? (
      <ComparisonReportDocument model={model} />
    ) : (
      <ReportDocument model={model} />
    )
  ) : (
    <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-foreground sm:text-sm">
      {report}
    </pre>
  );

  // Portaled to <body>, exactly like the Reports page print copy, so the
  // .report-print-root print rules apply to an identical DOM structure.
  return createPortal(
    <div className="report-print-root" data-report-ready="true" aria-hidden>
      {document_}
      <div className="report-print-footer justify-between border-t px-1 pt-2 text-[10px] text-muted-foreground">
        <span>
          EstateMind ·{" "}
          {comparison
            ? "Property Comparison Report"
            : "Property Investment Report"}
        </span>
        <span>Confidential — prepared for the client</span>
      </div>
    </div>,
    document.body
  );
}
