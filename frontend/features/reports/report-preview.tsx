"use client";

// Report preview + PDF export (Phase 15.17). Renders the structured report
// text returned by the backend as a premium document (report-parser.ts +
// report-document.tsx) and exports it as a PDF through the browser's
// print-to-PDF flow. All report content is produced by the backend — this
// component only presents it. When the text does not match the structured
// report layout (e.g. the non-enhanced fallback report) the raw text is shown
// instead, so the preview never breaks.

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { FileDown, Info } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ReportDocument } from "./report-document";
import { parseReport } from "./report-parser";

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
  useEffect(() => setMounted(true), []);

  const handleExportPdf = () => window.print();

  const document_ = model ? (
    <ReportDocument model={model} />
  ) : (
    <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-foreground sm:text-sm">
      {report}
    </pre>
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-muted-foreground">
          Report Preview
        </h2>
        <Button variant="outline" size="sm" onClick={handleExportPdf}>
          <FileDown /> Export PDF
        </Button>
      </div>

      {notice && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
          <Info className="mt-0.5 size-4 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      <div className="rounded-xl border bg-muted/40 p-3 sm:p-6">
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
