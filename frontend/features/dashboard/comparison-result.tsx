"use client";

// Comparison result renderer. Shows the winning property, its verdict and
// justification, then the full ranking table — mirroring the Streamlit
// render_comparison_result output. Premium compact presentation (Phase 15.18):
// a slim green winner strip (score + verdict inline) and the report-document
// ranking table with bg-primary/5 header + zebra rows. Every value is
// backend-verbatim.

import { Check, Medal, Trophy } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { ComparisonResult as ComparisonResultType } from "@/types/dashboard";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import { splitList } from "./format";
import { useWorkspace } from "./workspace-provider";

export function ComparisonResult({ data }: { data: ComparisonResultType }) {
  const { winner, rankings } = data;
  // Decision-first winner headline (Phase 15.20): show the human-readable
  // project name from the workspace's already-loaded backend search rows (the
  // same lookup CompactPropertyHeader uses); the raw id stays as metadata.
  const { properties } = useWorkspace();
  const nameOf = (id: string) =>
    properties.find((p) => p.id === id)?.project_name;
  const winnerName = nameOf(winner.id);

  // Executive polish (Phase 15.21): the backend comparison_reason is a
  // comma-separated justification — splitting it into ✓ advantages is
  // formatting only. The runner-up is the best-ranked non-winner row from the
  // backend's own ranking order.
  const advantages = splitList(winner.comparison_reason);
  const runnerUp = rankings.find((row) => row.id !== winner.id);
  const runnerUpName = runnerUp ? nameOf(runnerUp.id) : undefined;

  return (
    <div className="space-y-3">
      <div
        className={cn(
          "grid gap-3",
          runnerUp && "lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]"
        )}
      >
        {/* Winner card — large green banner: name, score, verdict, then the
            backend justification as ✓ advantages. */}
        <header className="overflow-hidden rounded-xl bg-primary text-primary-foreground shadow-sm">
          <div className="p-4">
            <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
              <Trophy className="size-3.5" /> Winner
            </p>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1.5">
              <h3 className="min-w-0 break-words font-heading text-2xl font-bold leading-tight sm:text-3xl">
                {winnerName ?? winner.id}
              </h3>
              {winnerName && (
                <span className="font-mono text-[10px] text-primary-foreground/60">
                  {winner.id}
                </span>
              )}
            </div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-primary-foreground/15 px-2.5 py-0.5 text-xs font-semibold">
                Score {winner.overall_score}
              </span>
              <span className="rounded-full bg-primary-foreground/15 px-2.5 py-0.5 text-xs font-semibold">
                {winner.verdict}
              </span>
            </div>
            {advantages.length > 1 ? (
              <ul className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1 border-t border-primary-foreground/15 pt-2.5">
                {advantages.map((advantage, i) => (
                  <li
                    key={i}
                    className="flex items-center gap-1.5 text-xs font-medium text-primary-foreground/90"
                  >
                    <Check className="size-3.5 shrink-0" />
                    {advantage}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1.5 text-sm leading-relaxed text-primary-foreground/85">
                {winner.comparison_reason}
              </p>
            )}
          </div>
        </header>

        {/* Runner-up card — the backend's next-ranked property. */}
        {runnerUp && (
          <div className="rounded-xl border bg-card p-3 shadow-sm">
            <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              <Medal className="size-3.5" /> Runner-up
            </p>
            <h4 className="mt-1 truncate font-heading text-base font-semibold">
              {runnerUpName ?? runnerUp.id}
            </h4>
            {runnerUpName && (
              <p className="truncate font-mono text-[10px] text-muted-foreground/70">
                {runnerUp.id}
              </p>
            )}
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <span className="font-heading text-xl font-semibold tabular-nums">
                {runnerUp.overall_score}
              </span>
              <StatusPill value={runnerUp.verdict} />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {runnerUp.comparison_reason}
            </p>
          </div>
        )}
      </div>

      {/* Ranking table — report-document table styling. */}
      <div className="overflow-hidden rounded-xl border shadow-sm">
        <Table>
          <TableHeader>
            <TableRow className="bg-primary/5 hover:bg-primary/5">
              <TableHead className="font-semibold text-foreground">
                Property ID
              </TableHead>
              <TableHead className="text-center font-semibold text-foreground">
                Score
              </TableHead>
              <TableHead className="text-center font-semibold text-foreground">
                Verdict
              </TableHead>
              <TableHead className="font-semibold text-foreground">
                Reason
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rankings.map((row, index) => {
              const isWinner = row.id === winner.id;
              return (
                <TableRow
                  key={row.id}
                  className={cn(
                    index % 2 === 1 && "bg-muted/40",
                    isWinner && "bg-primary/5 hover:bg-primary/5"
                  )}
                >
                  <TableCell className="font-mono text-xs">
                    <span className="flex items-center gap-1.5">
                      {isWinner && <Trophy className="size-3.5 text-primary" />}
                      {row.id}
                    </span>
                  </TableCell>
                  <TableCell
                    className={cn(
                      "text-center font-medium tabular-nums",
                      isWinner && "text-primary"
                    )}
                  >
                    {row.overall_score}
                  </TableCell>
                  <TableCell className="text-center">
                    <StatusPill value={row.verdict} />
                  </TableCell>
                  <TableCell className="max-w-80 text-xs text-muted-foreground">
                    {row.comparison_reason}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
