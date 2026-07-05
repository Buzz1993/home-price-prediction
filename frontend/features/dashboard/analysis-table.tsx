"use client";

// Generic table for the tabular analysis payloads (rental, prediction,
// valuation). Columns are derived from the backend records so the frontend
// stays agnostic to the exact metrics returned. Known price columns are shown
// in ₹ Cr to match the Streamlit formatting.

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { AnalysisRow } from "@/types/dashboard";
import { formatCell, formatCr, humanizeKey } from "./format";

const PRICE_COLUMNS = new Set([
  "price",
  "original_price",
  "predicted_price",
  "margin_diff",
]);

export function AnalysisTable({ rows }: { rows: AnalysisRow[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No data available.</p>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((col) => (
              <TableHead key={col}>{humanizeKey(col)}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, i) => (
            <TableRow key={i}>
              {columns.map((col) => (
                <TableCell key={col} className="whitespace-nowrap">
                  {PRICE_COLUMNS.has(col)
                    ? formatCr(row[col])
                    : formatCell(row[col])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
