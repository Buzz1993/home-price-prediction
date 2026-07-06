// Generated reports list for the Profile page. Presentational only — renders each
// report returned by GET /reports as a row with its title, optional summary, the
// properties it covers, and when it was created.

import { FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { ReportSummary } from "@/types/profile";

export function ReportsList({ reports }: { reports: ReportSummary[] }) {
  return (
    <ul className="divide-y">
      {reports.map((report) => (
        <li
          key={report.id}
          className="flex items-start gap-3 py-3 first:pt-0 last:pb-0"
        >
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
            <FileText className="size-4 text-muted-foreground" />
          </div>
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <p className="truncate text-sm font-medium">
                {report.title ?? `Report ${report.id}`}
              </p>
              {report.created_at && (
                <span className="shrink-0 text-xs text-muted-foreground">
                  {report.created_at}
                </span>
              )}
            </div>
            {report.summary && (
              <p className="line-clamp-2 text-sm text-muted-foreground">
                {report.summary}
              </p>
            )}
            {report.property_ids && report.property_ids.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-0.5">
                {report.property_ids.map((id) => (
                  <Badge key={id} variant="outline" className="font-mono">
                    {id}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
