"use client";

// Report preview + download. Renders the AI report text returned by the backend
// and lets the user download it as a Markdown file (client-side Blob — no
// business logic, the report content is produced entirely by the backend).

import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";

export function ReportPreview({ report }: { report: string }) {
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
      <div className="max-h-[24rem] overflow-y-auto rounded-lg border bg-muted/30 p-4">
        <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-foreground">
          {report}
        </pre>
      </div>
    </div>
  );
}
