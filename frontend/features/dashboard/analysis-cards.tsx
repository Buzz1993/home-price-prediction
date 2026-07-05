"use client";

// Card renderers for the richer, per-property analysis payloads: negotiation
// strategy and investment advisor. Mirrors the Streamlit expander/columns
// layout in render_extended_data_grids.

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { AdvisorRow, NegotiationRow } from "@/types/dashboard";
import { formatCr } from "./format";

export function NegotiationCards({ rows }: { rows: NegotiationRow[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {rows.map((item) => (
        <Card key={item.id} className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className="text-sm">
              <span className="font-mono">{item.id}</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 px-4 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Target price</span>
              <span className="font-medium">{formatCr(item.target_price)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Suggested discount</span>
              <span className="font-medium">
                {item.suggested_discount_percent}
              </span>
            </div>
            <div>
              <Badge variant="secondary">
                Leverage: {item.negotiation_power}
              </Badge>
            </div>
            <p>
              <span className="font-medium">Strategy: </span>
              {item.strategy}
            </p>
            <p className="rounded-md bg-muted p-2 text-muted-foreground">
              {item.talking_points}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function AdvisorCards({ rows }: { rows: AdvisorRow[] }) {
  return (
    <div className="space-y-3">
      {rows.map((item) => (
        <Card key={item.id} className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className="text-sm">
              <span className="font-mono">{item.id}</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 px-4 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="warning">For: {item.suitable_for}</Badge>
              <Badge variant="outline">{item.verdict}</Badge>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-md bg-emerald-500/5 p-2">
                <p className="mb-1 font-medium text-emerald-600 dark:text-emerald-400">
                  Positives
                </p>
                <p className="text-muted-foreground">{item.positives}</p>
              </div>
              <div className="rounded-md bg-destructive/5 p-2">
                <p className="mb-1 font-medium text-destructive">Risks</p>
                <p className="text-muted-foreground">{item.risks}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
