"use client";

// Report mutations for the Reports page. Two thin mutations — generate the AI
// report (POST /report) and share it to a phone number (POST /report/share).
// All report composition and delivery stays in the backend; these only track
// the request state, mirroring the Streamlit report workflow (generate → share).
//
// Phase 18.9: on a successful generation the finished report is recorded in the
// local Report History (report-history.ts) so it can be reopened, downloaded,
// shared and deleted later WITHOUT regenerating. Recording is a side effect
// only — the generation request and response are untouched.

import { useMutation } from "@tanstack/react-query";

import {
  generateReport,
  shareComparisonReport,
  shareReport,
} from "@/services/report-service";
import { addReport } from "./report-history";

export function useGenerateReport() {
  return useMutation({
    mutationFn: generateReport,
    onSuccess: (data, propertyIds) => {
      // Store the SAME text the preview shows (Claude-enhanced when available,
      // otherwise the unchanged backend report).
      addReport({
        type: "property",
        propertyIds,
        content: data.ai_enhanced ?? data.content,
        aiEnhanced: Boolean(data.ai_enhanced),
      });
    },
  });
}

export function useShareReport() {
  return useMutation({ mutationFn: shareReport });
}

// Share a stored COMPARISON report from the Report History — reuses the
// dedicated comparison share endpoint (Phase 17.3), same contract as the
// /compare page's own share flow. The stored report text is passed along so
// the backend delivers exactly that report without regenerating it.
export function useShareComparisonReport() {
  return useMutation({ mutationFn: shareComparisonReport });
}
