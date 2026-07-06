"use client";

// Report sharing. Collects a phone number and sends the generated report through
// POST /report/share, then shows the delivery status. Delivery itself is handled
// entirely by the backend (MCP tool + n8n workflow); this only captures the
// phone number and renders the returned status. Reproduces the Streamlit share
// step of the report workflow (enter phone → send → delivery status).

import { useState } from "react";
import { CheckCircle2, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { useShareReport } from "./use-report";

export function ShareReportForm({ report }: { report: string }) {
  const [phone, setPhone] = useState("");
  const share = useShareReport();

  const trimmed = phone.trim();
  const canShare = trimmed.length > 0 && !share.isPending;

  const handleShare = () => {
    if (!canShare) return;
    share.mutate({ phone_number: trimmed, report });
  };

  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="space-y-1">
        <h2 className="text-sm font-semibold">Share Report</h2>
        <p className="text-xs text-muted-foreground">
          Send this report to a phone number via the EstateMind delivery workflow.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="report-phone">Phone Number</Label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            id="report-phone"
            type="tel"
            inputMode="tel"
            placeholder="e.g. +91 98765 43210"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            className="sm:flex-1"
          />
          <Button onClick={handleShare} disabled={!canShare}>
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
      </div>

      {share.isSuccess && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-400">
          <CheckCircle2 className="size-4 shrink-0" />
          <span>Report sent successfully to {trimmed}.</span>
        </div>
      )}

      {share.isError && (
        <ErrorState
          title="Delivery failed"
          description="Something went wrong while sending the report. Please check the number and try again."
          onRetry={canShare ? handleShare : undefined}
          retrying={share.isPending}
        />
      )}
    </div>
  );
}
