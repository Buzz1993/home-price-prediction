"use client";

// Report preview + download. Renders the AI report returned by the backend and
// lets the user download it as a Markdown file (client-side Blob — no business
// logic, the report content is produced entirely by the backend).
//
// Phase 15.10: the displayed narrative is Claude's more readable re-presentation
// of the SAME backend report when available; when the AI enhancement is
// unavailable the original backend report is shown instead, with a friendly
// notice. Download always saves the displayed report.

import { Download, Info } from "lucide-react";

import { Button } from "@/components/ui/button";

export function ReportPreview({
  report,
  notice,
}: {
  report: string;
  notice?: string | null;
}) {
  const handleDownload = () => {
    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "estatemind-report.md";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-muted-foreground">
          Report Preview
        </h2>
        <Button variant="outline" size="sm" onClick={handleDownload}>
          <Download /> Download
        </Button>
      </div>

      {notice && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
          <Info className="mt-0.5 size-4 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      <div className="max-h-[24rem] overflow-y-auto rounded-lg border bg-muted/30 p-4">
        <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-foreground">
          {report}
        </pre>
      </div>
    </div>
  );
}
