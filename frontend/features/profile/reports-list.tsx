// Generated reports list for the Profile page. Presentational only — renders each
// report returned by GET /reports as a row with its title, optional summary, the
// properties it covers, and when it was created.
//
// Phase 18.7: premium report cards with better visual hierarchy, hover effects.

import { Calendar, FileText, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { ReportSummary } from "@/types/profile";

export function ReportsList({ reports }: { reports: ReportSummary[] }) {
  return (
    <ul className="space-y-3">
      {reports.map((report) => (
        <li
          key={report.id}
          className="group rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float"
        >
          <div className="flex items-start gap-4">
            <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary transition-colors duration-200 group-hover:bg-primary/20">
              <FileText className="size-6" />
            </div>
            <div className="min-w-0 flex-1 space-y-2">
              <div className="flex items-start justify-between gap-3">
                <p className="font-semibold leading-tight">
                  {report.title ?? `Report ${report.id}`}
                </p>
                {report.created_at && (
                  <div className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
                    <Calendar className="size-3.5" />
                    <span>{report.created_at}</span>
                  </div>
                )}
              </div>
              {report.summary && (
                <p className="line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                  {report.summary}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2">
                {report.property_ids && report.property_ids.length > 0 && (
                  <>
                    <Badge variant="secondary" className="gap-1 px-2 py-0.5">
                      <Sparkles className="size-3" />
                      {report.property_ids.length}{" "}
                      {report.property_ids.length === 1 ? "Property" : "Properties"}
                    </Badge>
                    {report.property_ids.slice(0, 3).map((id) => (
                      <Badge key={id} variant="outline" className="px-2 py-0.5 font-mono text-xs">
                        {id}
                      </Badge>
                    ))}
                    {report.property_ids.length > 3 && (
                      <Badge variant="outline" className="px-2 py-0.5 text-xs">
                        +{report.property_ids.length - 3} more
                      </Badge>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}
