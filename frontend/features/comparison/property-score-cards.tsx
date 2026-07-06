"use client";

// Property Score Cards (UI doc §9). A card per compared property showing its
// overall score, verdict and justification, with the winner highlighted. Purely
// presentational — the scores and verdicts come from the backend comparison.

import { Trophy } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ComparisonRow } from "@/types/dashboard";

export function PropertyScoreCards({
  rankings,
  winnerId,
}: {
  rankings: ComparisonRow[];
  winnerId: string;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {rankings.map((row) => {
        const isWinner = row.id === winnerId;
        return (
          <Card
            key={row.id}
            className={cn(
              "gap-0 py-0",
              isWinner && "border-emerald-500/50 ring-1 ring-emerald-500/40"
            )}
          >
            <CardContent className="space-y-3 p-4">
              <div className="flex items-start justify-between gap-2">
                <span className="min-w-0 truncate font-mono text-xs text-muted-foreground">
                  {row.id}
                </span>
                {isWinner && (
                  <Badge variant="success" className="shrink-0">
                    <Trophy /> Winner
                  </Badge>
                )}
              </div>

              <div className="flex items-baseline gap-1.5">
                <span className="text-3xl font-semibold tabular-nums">
                  {row.overall_score}
                </span>
                <span className="text-xs text-muted-foreground">score</span>
              </div>

              <Badge variant="secondary">{row.verdict}</Badge>

              <p className="text-xs text-muted-foreground">
                {row.comparison_reason}
              </p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
