"use client";

// Comparison Report persistence (Phase 18.15). One shared, once-per-comparison
// source of the Comparison Report text: the FIRST caller triggers the existing
// POST /report/comparison endpoint and records the finished report in the
// SAME local Report History the Reports page uses (report-history.ts);
// every later caller for the same compared ids awaits that same promise.
//
// The comparison workspace calls this in the background as soon as a
// comparison succeeds (so the report lands in Recent Reports automatically),
// and the export/share toolbar reuses the identical promise — the report is
// never generated twice and never recorded twice. No backend, API or report
// generation logic changes; this only sequences the existing call.

import { useCallback, useRef } from "react";

import { generateComparisonReport } from "@/services/report-service";
import { addReport } from "@/features/reports/report-history";

export function useComparisonReport() {
  // The in-flight (or finished) generation, keyed by the compared ids so a
  // new comparison generates a new report while re-runs reuse the old one.
  const current = useRef<{ key: string; promise: Promise<string> } | null>(
    null
  );

  return useCallback((ids: string[]): Promise<string> => {
    const key = ids.join("|");
    if (current.current?.key === key) return current.current.promise;

    const promise = generateComparisonReport(ids).then((generated) => {
      // Store the SAME text the preview shows (Claude-enhanced when
      // available, otherwise the unchanged backend report) — the exact rule
      // the Reports page follows in use-report.ts.
      const text = generated.ai_enhanced ?? generated.content;
      addReport({
        type: "comparison",
        propertyIds: ids,
        content: text,
        aiEnhanced: Boolean(generated.ai_enhanced),
      });
      return text;
    });
    // A failed generation frees the slot so the next export/share can retry.
    promise.catch(() => {
      if (current.current?.key === key) current.current = null;
    });

    current.current = { key, promise };
    return promise;
  }, []);
}
