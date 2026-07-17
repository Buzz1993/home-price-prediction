"use client";

// Report sharing. Collects a phone number and sends the generated report through
// POST /report/share, then shows the delivery status. Delivery itself is handled
// entirely by the backend, which renders the report to a PDF and sends it via
// the official Meta WhatsApp Cloud API (Phase 16.1). The previewed report text
// is passed along so the backend delivers exactly what the user sees without
// regenerating it. Phase 17.5: the send button, phone input and helper text all
// switch to the sending state in the same render frame as the click, duplicate
// submissions are blocked, and success/failure banners restate the delivery
// number — presentation only, the share flow itself is unchanged.

import { useState } from "react";
import { CheckCircle2, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "@/lib/api-client";
import { useShareComparisonReport, useShareReport } from "./use-report";
import type { ReportType } from "./report-history";

export function ShareReportForm({
  propertyIds,
  report,
  type = "property",
}: {
  propertyIds: string[];
  // The previewed report text (Phase 16.1) — sent as-is, never regenerated.
  report?: string;
  // Which existing share endpoint to use (Phase 18.9 report history):
  // "property" → POST /report/share, "comparison" → POST /report/comparison/share.
  type?: ReportType;
}) {
  const [phone, setPhone] = useState("");
  const shareProperty = useShareReport();
  const shareComparison = useShareComparisonReport();
  const share = type === "comparison" ? shareComparison : shareProperty;

  const trimmed = phone.trim();
  const canShare = trimmed.length > 0 && !share.isPending;

  const handleShare = () => {
    if (!canShare) return;
    share.mutate({
      property_ids: propertyIds,
      phone_number: trimmed,
      report,
    });
  };

  // Report the delivery faithfully from the backend's returned ShareResult
  // rather than assuming any HTTP 2xx means the report was delivered.
  const result = share.data;
  const delivered =
    share.isSuccess &&
    (result?.status_code === undefined ||
      (result.status_code >= 200 && result.status_code < 300));
  const failed = share.isError || (share.isSuccess && !delivered);
  // The backend returns meaningful error details (invalid number, expired
  // token, upload failure, timeout, missing config) — surface them.
  const failureDetail =
    share.error instanceof ApiError ? share.error.message : undefined;
  // Show the number the report was actually sent to, not the (possibly edited)
  // current input value.
  const sentPhone = share.variables?.phone_number ?? trimmed;

  return (
    <div className="space-y-4 rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-5 shadow-float">
      <div className="space-y-1.5">
        <h2 className="text-base font-semibold">Share Report</h2>
        <p className="text-sm text-muted-foreground">
          Send this comprehensive report as a PDF to WhatsApp
        </p>
      </div>

      <div className="space-y-3">
        <Label htmlFor="report-phone" className="text-sm font-medium">
          WhatsApp Number
        </Label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <Input
            id="report-phone"
            type="tel"
            inputMode="tel"
            placeholder="e.g. +91 98765 43210"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            disabled={share.isPending}
            className="sm:flex-1"
          />
          <Button onClick={handleShare} disabled={!canShare} className="gap-2">
            {share.isPending ? (
              <>
                <Spinner /> Sending…
              </>
            ) : (
              <>
                <Send /> Share Report
              </>
            )}
          </Button>
        </div>
        {share.isPending && (
          <p className="text-sm text-muted-foreground">
            Sending property report to WhatsApp…
          </p>
        )}
      </div>

      {delivered && (
        <div className="flex items-start gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm text-emerald-700 dark:text-emerald-400">
          <CheckCircle2 className="mt-0.5 size-5 shrink-0" />
          <div className="space-y-1">
            <p className="font-semibold">Report sent successfully</p>
            <p className="text-xs text-emerald-700/80 dark:text-emerald-400/80">
              Delivered to {sentPhone}
            </p>
          </div>
        </div>
      )}

      {failed && (
        <ErrorState
          title="Unable to send report"
          description={
            failureDetail ??
            "Something went wrong while sending the report. Please check the number and try again."
          }
          onRetry={canShare ? handleShare : undefined}
          retrying={share.isPending}
        />
      )}
    </div>
  );
}
