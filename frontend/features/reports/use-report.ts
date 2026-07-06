"use client";

// Report mutations for the Reports page. Two thin mutations — generate the AI
// report (POST /report) and share it to a phone number (POST /report/share).
// All report composition and delivery stays in the backend; these only track
// the request state, mirroring the Streamlit report workflow (generate → share).

import { useMutation } from "@tanstack/react-query";

import { generateReport, shareReport } from "@/services/report-service";

export function useGenerateReport() {
  return useMutation({ mutationFn: generateReport });
}

export function useShareReport() {
  return useMutation({ mutationFn: shareReport });
}
