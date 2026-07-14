"use client";

// Future Growth view (Phase 15.15). The backend has no dedicated future-growth
// endpoint — run_future_agent computes the growth fields during enrichment and
// they are exposed on the investment advice rows (get_investment_advice /
// /analysis/advisor: growth_label, growth_reason, and — when the backend selects
// them — future_signals, infra_detected, growth_score). So Future Growth is
// surfaced from those advisor rows here, focused on the growth dimension.
// Presentational only: every value shown is produced by the backend.

import { LineChart, Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";
import type { AdvisorRow } from "@/types/dashboard";

// Map the backend growth label to a badge tone. The backend labels are e.g.
// "🚀 High Growth", "📍 Mature Area", "➖ No Growth Signal"; we only pick a colour
// from the wording and never alter the text the backend produced.
function growthTone(label: string): "success" | "warning" | "outline" {
  const text = label.toLowerCase();
  if (text.includes("high")) return "success";
  if (text.includes("mature")) return "warning";
  return "outline";
}

// A backend growth field is "present" only when it is a non-empty value. The
// advisor rows return null/empty for fields the backend did not populate, so we
// render each detail row only when there is something to show.
function hasValue(value: unknown): value is string | number {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

function GrowthDetail({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

export function FutureGrowthCards({ rows }: { rows: AdvisorRow[] }) {
  return (
    <div className="space-y-3">
      {rows.map((item) => {
        // Growth data present for this property? growth_label is the primary
        // signal produced by the backend future agent.
        const hasGrowth =
          hasValue(item.growth_label) ||
          hasValue(item.growth_reason) ||
          hasValue(item.future_signals) ||
          hasValue(item.infra_detected) ||
          hasValue(item.growth_score);

        return (
          <Card key={item.id} className="gap-3 py-4">
            <CardHeader className="px-4">
              <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                <LineChart className="size-4 text-primary" />
                <span className="font-mono">{item.id}</span>
                {hasValue(item.growth_label) && (
                  <Badge variant={growthTone(item.growth_label as string)}>
                    {item.growth_label}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 px-4 text-sm">
              {hasGrowth ? (
                <>
                  {hasValue(item.growth_reason) && (
                    <div className="rounded-md bg-emerald-500/5 p-3">
                      <p className="mb-1 font-medium text-emerald-600 dark:text-emerald-400">
                        Growth outlook
                      </p>
                      <p className="text-muted-foreground">
                        {item.growth_reason}
                      </p>
                    </div>
                  )}
                  {(hasValue(item.future_signals) ||
                    hasValue(item.infra_detected) ||
                    hasValue(item.growth_score)) && (
                    <div className="space-y-2 rounded-md bg-muted p-3">
                      {hasValue(item.future_signals) && (
                        <GrowthDetail
                          label="Future signals"
                          value={item.future_signals as string}
                        />
                      )}
                      {hasValue(item.infra_detected) && (
                        <GrowthDetail
                          label="Infrastructure detected"
                          value={item.infra_detected as string}
                        />
                      )}
                      {hasValue(item.growth_score) && (
                        <GrowthDetail
                          label="Growth score"
                          value={item.growth_score as number}
                        />
                      )}
                    </div>
                  )}
                </>
              ) : (
                <EmptyState
                  icon={Sparkles}
                  description="No future growth information is available for this property."
                  className="py-6"
                />
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
