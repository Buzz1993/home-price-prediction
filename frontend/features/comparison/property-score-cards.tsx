"use client";

// Property Score Cards (UI doc §9). A card per compared property showing its
// overall score, verdict and justification, with the winner highlighted. Purely
// presentational — the scores and verdicts come from the backend comparison.
// Premium presentation (Phase 15.18): report-style cards with a score bar whose
// width is proportional to the highest backend score (visual only — the raw
// backend score is what is displayed).

import { Trophy } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ComparisonRow } from "@/types/dashboard";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";

export function PropertyScoreCards({
  rankings,
  winnerId,
}: {
  rankings: ComparisonRow[];
  winnerId: string;
}) {
  // Scale bars against the best score so the cards read as a visual ranking.
  const maxScore = Math.max(
    ...rankings.map((row) => Number(row.overall_score) || 0),
    0
  );

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {rankings.map((row) => {
        const isWinner = row.id === winnerId;
        const score = Number(row.overall_score) || 0;
        const percent =
          maxScore > 0 ? Math.max(0, Math.min(100, (score / maxScore) * 100)) : 0;

        return (
          <div
            key={row.id}
            className={cn(
              "overflow-hidden rounded-xl border bg-card shadow-sm transition-shadow hover:shadow-md",
              isWinner && "border-primary/50 ring-1 ring-primary/30"
            )}
          >
            {/* Winner gets the report document's brand accent bar. */}
            <div className={cn("h-1", isWinner ? "bg-primary" : "bg-border")} />
            <div className="space-y-2 p-3">
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

              <div className="flex items-center gap-2.5">
                <span
                  className={cn(
                    "font-heading text-2xl font-semibold tabular-nums",
                    isWinner && "text-primary"
                  )}
                >
                  {row.overall_score}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    score
                  </p>
                  <div className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className={cn(
                        "h-full rounded-full",
                        isWinner ? "bg-primary" : "bg-muted-foreground/40"
                      )}
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                </div>
                <StatusPill value={row.verdict} className="shrink-0" />
              </div>

              <p className="text-xs leading-relaxed text-muted-foreground">
                {row.comparison_reason}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
