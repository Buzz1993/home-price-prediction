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
import type { EnhancedReport, ShareResult } from "@/types/dashboard";

// POST /report?enhance=true — generate an AI property report for the selected
// properties and additionally ask Claude to improve its readability (Phase
// 15.10). The backend still composes the report; the enhancement only
// re-presents that SAME report. The response is
// `{ content, ai_enhanced }` — `content` is the unchanged backend report and
// `ai_enhanced` is the polished version (or null when the AI enhancement is
// unavailable, so the backend report always renders).
export function generateReport(ids: string[]): Promise<EnhancedReport> {
  return apiRequest<EnhancedReport>("/report?enhance=true", {
    method: "POST",
    body: { property_ids: ids },
  });
}

// POST /report/share?enhance=true — share a report for the selected properties
// to a phone number. The backend (ShareReportRequest in src/api/report_api.py)
// regenerates the report from the property ids, enhances its readability with
// Claude (Phase 15.10) so the shared report matches the enhanced preview, and
// forwards it to the n8n workflow (send_property_report), so the request carries
// the ids, not the report text.
export function shareReport(payload: {
  property_ids: string[];
  phone_number: string;
}): Promise<ShareResult> {
  return apiRequest<ShareResult>("/report/share?enhance=true", {
    method: "POST",
    body: payload,
  });
}
