// Report API calls. Thin wrappers over the documented Report endpoints
// (POST /report, POST /report/share — see project_docs/03_API.md). The backend
// owns report composition and delivery: report generation is assembled from the
// existing analysis services, and delivery (Phase 16.1) renders the report to a
// PDF and sends it through the official Meta WhatsApp Cloud API
// (src/services/whatsapp_service.py). The frontend only sends the request and
// renders the response.

import { API_BASE_URL, ApiError, apiRequest } from "@/lib/api-client";
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

// POST /report/share?enhance=true — send a report to a WhatsApp number.
// `report` carries the ALREADY-GENERATED report text the user is previewing
// (Phase 16.1), so the backend delivers exactly that report as a PDF without
// regenerating it. When `report` is absent the backend generates (and, with
// enhance=true, Claude-polishes) the report once before sending — the
// pre-16.1 contract.
export function shareReport(payload: {
  property_ids: string[];
  phone_number: string;
  report?: string;
}): Promise<ShareResult> {
  return apiRequest<ShareResult>("/report/share?enhance=true", {
    method: "POST",
    body: payload,
  });
}

// POST /report/pdf — download the previewed report as a PDF (Phase 16.2).
// The backend's single Chromium-based generator (src/services/pdf_service.py)
// prints the existing report page, so this PDF is pixel-identical to the
// WhatsApp document and the browser's print output. Returned as a Blob
// because the response is a binary file, not JSON.
export async function downloadReportPdf(report: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/report/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report }),
  });

  if (!response.ok) {
    let detail = response.statusText || "Request failed";
    try {
      const data = await response.json();
      if (typeof data?.detail === "string") detail = data.detail;
    } catch {
      // Non-JSON error body — keep the status text.
    }
    throw new ApiError(detail, response.status);
  }

  return response.blob();
}
