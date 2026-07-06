// Report API calls. Thin wrappers over the documented Report endpoints
// (POST /report, POST /report/share — see project_docs/03_API.md). The backend
// owns report composition and delivery: report generation is assembled from the
// existing analysis services, and delivery runs through the MCP tool
// send_property_report (src/mcp/tools/property_tools.py) which posts to the n8n
// workflow. The frontend only sends the request and renders the response.
//
// Backend limitation: the EstateMind Copilot API (src/api) currently exposes
// only the /analysis/* endpoints — neither /report nor /report/share is wired
// yet. These wrappers target the documented contract so live data flows once the
// endpoints are exposed, matching the Phase 4–8 pattern. No API is invented.

import { apiRequest } from "@/lib/api-client";
import type { ReportResult, ShareResult } from "@/types/dashboard";

// POST /report — generate an AI property report for the selected properties.
export function generateReport(ids: string[]): Promise<ReportResult> {
  return apiRequest<ReportResult>("/report", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /report/share — send a generated report to a phone number. The backend
// forwards it to the n8n workflow (send_property_report expects phone + report).
export function shareReport(payload: {
  phone_number: string;
  report: string;
}): Promise<ShareResult> {
  return apiRequest<ShareResult>("/report/share", {
    method: "POST",
    body: payload,
  });
}
