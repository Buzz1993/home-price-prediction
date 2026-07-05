"use client";

// Comparison result renderer. Shows the winning property, its verdict and
// justification, then the full ranking table — mirroring the Streamlit
// render_comparison_result output.

import { Trophy } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import type { ComparisonResult as ComparisonResultType } from "@/types/dashboard";

export function ComparisonResult({ data }: { data: ComparisonResultType }) {
  const { winner, rankings } = data;

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-emerald-500/5 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="success">
            <Trophy /> Winner
          </Badge>
          <span className="font-mono text-sm font-medium">{winner.id}</span>
          <span className="text-sm text-muted-foreground">
            Score {winner.overall_score}
          </span>
        </div>
        <p className="mt-2 text-sm font-medium">{winner.verdict}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          {winner.comparison_reason}
        </p>
      </div>

      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Property ID</TableHead>
              <TableHead>Score</TableHead>
              <TableHead>Verdict</TableHead>
              <TableHead>Reason</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rankings.map((row) => (
              <TableRow key={row.id}>
                <TableCell className="font-mono text-xs">{row.id}</TableCell>
                <TableCell className="font-medium">
                  {row.overall_score}
                </TableCell>
                <TableCell>{row.verdict}</TableCell>
                <TableCell className="max-w-80 text-xs text-muted-foreground">
                  {row.comparison_reason}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
