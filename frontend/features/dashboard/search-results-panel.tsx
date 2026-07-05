"use client";

// Search Results Panel. Renders ranked properties returned by the backend and
// lets the user stage them into the evaluation tray via checkboxes — the same
// interaction as the Streamlit search results data-editor.

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import type { SearchResult } from "@/types/dashboard";
import { useWorkspace } from "./workspace-provider";
import { formatCr } from "./format";

export function SearchResultsPanel({ results }: { results: SearchResult[] }) {
  const { tray, toggleTray } = useWorkspace();

  if (results.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No properties matched your search. Try adjusting the location, BHK or
        amenities.
      </p>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-10" />
            <TableHead>Property ID</TableHead>
            <TableHead>Price</TableHead>
            <TableHead>BHK</TableHead>
            <TableHead>Locality</TableHead>
            <TableHead>Amenities</TableHead>
            <TableHead>Why Recommended</TableHead>
            <TableHead>BM25 Score</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {results.map((r) => {
            const staged = tray.includes(r.id);
            return (
              <TableRow key={r.id} data-state={staged ? "selected" : undefined}>
                <TableCell>
                  <Checkbox
                    checked={staged}
                    onCheckedChange={() => toggleTray(r.id)}
                    aria-label={`Stage property ${r.id}`}
                  />
                </TableCell>
                <TableCell className="font-mono text-xs">{r.id}</TableCell>
                <TableCell className="whitespace-nowrap font-medium">
                  {formatCr(r.price)}
                </TableCell>
                <TableCell className="whitespace-nowrap">{r.bhk_type}</TableCell>
                <TableCell>{r.location}</TableCell>
                <TableCell className="max-w-40 truncate" title={r.amenities_mcp}>
                  {r.amenities_mcp || "—"}
                </TableCell>
                <TableCell className="max-w-72 text-xs text-muted-foreground">
                  {r.why_recommended}
                </TableCell>
                <TableCell className="whitespace-nowrap tabular-nums">
                  {r.search_score.toFixed(4)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
