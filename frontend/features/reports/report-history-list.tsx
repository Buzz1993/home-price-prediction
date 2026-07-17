"use client";

// Report History list (Phase 18.9). Renders the locally stored reports
// (report-history.ts) as premium cards — newest first — each with its icon,
// title, generation date/time, property count and an "AI Generated" status,
// plus Preview / Download PDF / Share WhatsApp / Delete actions.
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
  Sparkles,
  Trash2,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
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
        "rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float" +
        (active ? " border-primary/40 ring-1 ring-primary/20" : "")
      }
    >
      <div className="flex items-start gap-4">
        <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Icon className="size-6" />
        </div>
        <div className="min-w-0 flex-1 space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-semibold leading-tight">{report.title}</p>
            <Badge className="gap-1 px-2.5 py-0.5">
              <Sparkles className="size-3" /> AI Generated
            </Badge>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span>{generated.toLocaleDateString()}</span>
            <span aria-hidden>·</span>
            <span>
              {generated.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
            <span aria-hidden>·</span>
            <span>
              {report.propertyIds.length}{" "}
              {report.propertyIds.length === 1 ? "property" : "properties"}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-0.5">
            <Button size="sm" className="gap-1.5" onClick={() => onPreview(report)}>
              <Eye /> Preview
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? <Spinner /> : <FileDown />}
              {downloading ? "Downloading…" : "Download PDF"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => setShareOpen((open) => !open)}
            >
              <Send /> Share WhatsApp
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => onDelete(report.id)}
            >
              <Trash2 /> Delete
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
