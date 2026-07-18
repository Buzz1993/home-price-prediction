"use client";

// Report History list (Phase 18.9, redesigned in Phase 18.14). Renders the
// locally stored reports (report-history.ts) as compact cards — newest first —
// each with its icon, title, generation date/time and property ids, plus
// Preview / Download PDF / Share WhatsApp actions and a top-right Delete.
//
// Nothing is regenerated here: Preview reopens the stored report text,
// Download renders that SAME text through the existing single PDF endpoint
// (POST /report/pdf), and Share passes it to the existing share endpoints so
// the backend delivers exactly what was generated. Delete only removes the
// local history entry.

import { useState } from "react";
import {
  Eye,
  FileDown,
  FileText,
  Scale,
  Send,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { downloadReportPdf } from "@/services/report-service";
import type { StoredReport } from "./report-history";
import { ShareReportForm } from "./share-report-form";

export function ReportHistoryList({
  reports,
  activeId,
  onPreview,
  onDelete,
}: {
  reports: StoredReport[];
  // The report currently open in the preview (highlighted).
  activeId: string | null;
  onPreview: (report: StoredReport) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <ul className="space-y-3">
      {reports.map((report) => (
        <ReportHistoryCard
          key={report.id}
          report={report}
          active={report.id === activeId}
          onPreview={onPreview}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}

function ReportHistoryCard({
  report,
  active,
  onPreview,
  onDelete,
}: {
  report: StoredReport;
  active: boolean;
  onPreview: (report: StoredReport) => void;
  onDelete: (id: string) => void;
}) {
  const [shareOpen, setShareOpen] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const generated = new Date(report.createdAt);
  const Icon = report.type === "comparison" ? Scale : FileText;

  // Download the stored report through the existing PDF endpoint — identical
  // to the preview's Export PDF, with the browser print flow as fallback.
  const handleDownload = async () => {
    setDownloading(true);
    try {
      const blob = await downloadReportPdf(report.content);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download =
        report.type === "comparison"
          ? "EstateMind Property Comparison.pdf"
          : "EstateMind Investment Report.pdf";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      // Renderer unavailable — reopen the preview so print-to-PDF matches.
      onPreview(report);
      window.print();
    } finally {
      setDownloading(false);
    }
  };

  return (
    <li
      className={
        "rounded-xl border bg-card p-3 shadow-sm transition-all duration-200 hover:shadow-float" +
        (active ? " border-primary/40 ring-1 ring-primary/20" : "")
      }
    >
      <div className="flex items-center gap-4">
        <div className="flex size-11 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Icon className="size-5" />
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-start justify-between gap-2">
            <p className="truncate font-semibold leading-tight">{report.title}</p>
            <button
              type="button"
              className="inline-flex shrink-0 items-center gap-1 text-xs text-destructive transition-colors hover:text-destructive/80"
              onClick={() => onDelete(report.id)}
            >
              <Trash2 className="size-3.5" /> Delete
            </button>
          </div>
          <p className="text-xs text-muted-foreground">
            {generated.toLocaleDateString()}{" "}
            <span aria-hidden>•</span>{" "}
            {generated.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </p>
          {/* Property ids covered by this report — small monospace, light
              gray, stacked, max 3 with a "+N more". */}
          {report.propertyIds.length > 0 && (
            <div className="flex flex-col">
              {report.propertyIds.slice(0, 3).map((id) => (
                <span
                  key={id}
                  className="truncate font-mono text-[0.7rem] leading-snug text-muted-foreground/70"
                >
                  {id}
                </span>
              ))}
              {report.propertyIds.length > 3 && (
                <span className="font-mono text-[0.7rem] leading-snug text-muted-foreground/60">
                  +{report.propertyIds.length - 3} more
                </span>
              )}
            </div>
          )}
          <div className="flex items-center gap-2 pt-1.5">
            <Button size="sm" className="gap-1.5" onClick={() => onPreview(report)}>
              <Eye /> Preview
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5 bg-background"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? <Spinner /> : <FileDown />}
              {downloading ? "Downloading…" : "Download PDF"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5 bg-background"
              onClick={() => setShareOpen((open) => !open)}
            >
              <Send /> Share WhatsApp
            </Button>
          </div>
          {shareOpen && (
            <div className="pt-1">
              <ShareReportForm
                propertyIds={report.propertyIds}
                report={report.content}
                type={report.type}
              />
            </div>
          )}
        </div>
      </div>
    </li>
  );
}
