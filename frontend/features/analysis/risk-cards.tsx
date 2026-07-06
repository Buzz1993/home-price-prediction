"use client";

// Risk Analysis view. The backend has no dedicated risk endpoint or tool — it
// embeds risk metrics inside the investment advice (get_investment_advice, whose
// docstring notes it "charts risk metrics against investment horizons"). So Risk
// Analysis is surfaced from the advisor rows here, focused on the risk/verdict
// dimension. Presentational only.

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { AdvisorRow } from "@/types/dashboard";

export function RiskCards({ rows }: { rows: AdvisorRow[] }) {
  return (
    <div className="space-y-3">
      {rows.map((item) => (
        <Card key={item.id} className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-mono">{item.id}</span>
              <Badge variant="outline">{item.verdict}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 px-4 text-sm">
            <div className="rounded-md bg-destructive/5 p-3">
              <p className="mb-1 font-medium text-destructive">
                Identified risks
              </p>
              <p className="text-muted-foreground">{item.risks}</p>
            </div>
            <p className="text-xs text-muted-foreground">
              Best suited for {item.suitable_for} investors.
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
