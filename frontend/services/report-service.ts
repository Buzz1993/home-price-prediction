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
import type { ShareResult } from "@/types/dashboard";

// POST /report — generate an AI property report for the selected properties.
// The backend (create_property_report in src/mcp/tools/property_tools.py returns
// str) sends the markdown report back as a bare JSON string, so the response is
// the report text itself.
export function generateReport(ids: string[]): Promise<string> {
  return apiRequest<string>("/report", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /report/share — share a report for the selected properties to a phone
// number. The backend (ShareReportRequest in src/api/report_api.py) regenerates
// the report from the property ids and forwards it to the n8n workflow
// (send_property_report), so the request carries the ids, not the report text.
export function shareReport(payload: {
  property_ids: string[];
  phone_number: string;
}): Promise<ShareResult> {
  return apiRequest<ShareResult>("/report/share", {
    method: "POST",
    body: payload,
  });
}
