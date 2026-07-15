"use client";

// Generic comparison-matrix table for the Property Comparison workspace
// (Phase 17.0). Attributes run down the sticky first column, each compared
// property occupies one column, and the best / worst backend value per row is
// tinted soft emerald / soft rose with a trophy badge on the winner. Pure
// presentation: cell tones arrive precomputed by the callers from existing
// backend values (compare-utils) — nothing is scored or judged here.

import { Home, Info, Trophy, type LucideIcon } from "lucide-react";
import { useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { CellTone } from "./compare-utils";

// One compared property (a table column). `header` swaps the compact default
// header for a custom node (the Phase 17.1 premium header card); `isWinner`
// marks the overall backend comparison winner (trophy + subtle column tint);
// `verdict` is the property's verbatim backend advisor verdict, shown by the
// premium header card.
export type MatrixColumn = {
  id: string;
  name: string;
  sub?: string;
  imageUrl?: string;
  header?: React.ReactNode;
  isWinner?: boolean;
  verdict?: unknown;
};

// One value cell. `badge` labels the winning cell (e.g. "Lowest price");
// `barFraction` (0..1) draws the Goal-2 proportional value bar under the
// number; `note` is the Goal-6 relative-difference helper line. Both arrive
// precomputed from existing backend values (compare-utils).
export type MatrixCell = {
  content: React.ReactNode;
  tone?: CellTone;
  badge?: string;
  barFraction?: number | null;
  note?: string | null;
};

export type MatrixRowData = {
  label: string;
  tooltip?: string;
  cells: MatrixCell[];
};

const TONE_CELL: Record<CellTone, string> = {
  best: "bg-emerald-50/80 font-semibold text-emerald-900",
  worst: "bg-rose-50/60 text-rose-700",
  neutral: "",
};

// Small square thumbnail for the sticky property header (same plain-<img> +
// fallback approach as CompactPropertyHeader).
function ColumnThumbnail({ column }: { column: MatrixColumn }) {
  const [imageError, setImageError] = useState(false);
  const showImage = Boolean(column.imageUrl) && !imageError;

  return (
    <div className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-primary/10">
      {showImage ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={column.imageUrl}
          alt={column.name}
          loading="lazy"
          decoding="async"
          onError={() => setImageError(true)}
          className="size-full object-cover"
        />
      ) : (
        <Home className="size-4 text-primary" />
      )}
    </div>
  );
}

// Attribute label with the optional ⓘ explainer tooltip.
function RowLabel({ label, tooltip }: { label: string; tooltip?: string }) {
  return (
    <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
      {label}
      {tooltip && (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label={`What is ${label}?`}
              className="text-muted-foreground/50 transition-colors hover:text-muted-foreground"
            >
              <Info className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent>{tooltip}</TooltipContent>
        </Tooltip>
      )}
    </span>
  );
}

// One comparison table: sticky attribute column, sticky property header row,
// zebra data rows, horizontal scroll on small screens. `bare` drops the card
// chrome for embedding inside an accordion item (Phase 17.1), which provides
// its own border and title.
export function MatrixTable({
  title,
  icon: Icon,
  description,
  columns,
  rows,
  showThumbnails = false,
  bare = false,
}: {
  title?: string;
  icon?: LucideIcon;
  description?: string;
  columns: MatrixColumn[];
  rows: MatrixRowData[];
  showThumbnails?: boolean;
  bare?: boolean;
}) {
  if (rows.length === 0) return null;

  return (
    <section
      className={cn(
        !bare && "overflow-hidden rounded-xl border bg-card shadow-sm"
      )}
    >
      {title && (
        <div className="border-b p-3">
          <h3 className="flex items-center gap-2 font-heading text-sm font-semibold">
            {Icon && <Icon className="size-4 text-primary" />}
            {title}
          </h3>
          {description && (
            <p className="mt-0.5 text-xs text-muted-foreground">
              {description}
            </p>
          )}
        </div>
      )}

      <div className="max-h-[75vh] overflow-auto">
        <table className="w-full min-w-[40rem] text-sm">
          <thead className="sticky top-0 z-20">
            <tr className="border-b">
              {/* Sticky corner cell — above both scroll directions. */}
              <th className="sticky left-0 z-10 w-44 min-w-44 border-r bg-emerald-50 px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                Attribute
              </th>
              {columns.map((column) => (
                <th
                  key={column.id}
                  className={cn(
                    "min-w-40 bg-emerald-50 px-3 py-2 text-left align-bottom",
                    column.isWinner && "bg-emerald-100/80"
                  )}
                >
                  {column.header ?? (
                    <div className="flex items-center gap-2">
                      {showThumbnails && <ColumnThumbnail column={column} />}
                      <div className="min-w-0">
                        <p className="flex items-center gap-1.5 truncate font-heading text-sm font-semibold text-foreground">
                          {column.isWinner && (
                            <Trophy className="size-3.5 shrink-0 text-primary" />
                          )}
                          <span className="min-w-0 truncate">
                            {column.name}
                          </span>
                        </p>
                        {column.sub && (
                          <p className="truncate text-[11px] font-normal normal-case text-muted-foreground">
                            {column.sub}
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr
                key={`${row.label}-${rowIndex}`}
                className="border-b transition-colors last:border-0 hover:bg-muted/40"
              >
                {/* The attribute column stays readable while the property
                    columns scroll horizontally beneath it. */}
                <th
                  scope="row"
                  className="sticky left-0 z-10 border-r bg-card px-3 py-2.5 text-left align-middle"
                >
                  <RowLabel label={row.label} tooltip={row.tooltip} />
                </th>
                {row.cells.map((cell, cellIndex) => {
                  const tone = cell.tone ?? "neutral";
                  return (
                    <td
                      key={cellIndex}
                      className={cn(
                        "px-3 py-2.5 align-middle",
                        rowIndex % 2 === 1 && tone === "neutral" && "bg-muted/20",
                        // Subtle full-column tint under the overall winner.
                        columns[cellIndex]?.isWinner &&
                          tone === "neutral" &&
                          "bg-primary/5",
                        TONE_CELL[tone]
                      )}
                    >
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="min-w-0 break-words">
                          {cell.content}
                        </span>
                        {tone === "best" && cell.badge && (
                          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
                            <Trophy className="size-3" />
                            {cell.badge}
                          </span>
                        )}
                      </div>
                      {/* Goal 2 — proportional value bar under the number so
                          rows compare at a glance without reading values. */}
                      {typeof cell.barFraction === "number" && (
                        <div className="mt-1.5 h-1.5 w-full max-w-40 overflow-hidden rounded-full bg-muted">
                          <div
                            className={cn(
                              "h-full rounded-full transition-[width] duration-700 ease-out",
                              tone === "best" ? "bg-primary" : "bg-primary/30"
                            )}
                            style={{ width: `${cell.barFraction * 100}%` }}
                          />
                        </div>
                      )}
                      {/* Goal 6 — relative difference vs. the other compared
                          properties (display arithmetic on backend values). */}
                      {cell.note && (
                        <p className="mt-1 max-w-40 text-[10px] leading-snug text-emerald-700/90">
                          {cell.note}
                        </p>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// "Why this property won" strip closing each analysis section (Phase 17.1,
// Goal 5): trophy + category, the winner's name, then an explicit "Why?"
// ✓ checklist. Every point is a backend-derived reason precomputed by
// compare-winners — no AI call, no new judgement.
export function SectionWinnerCard({
  category,
  name,
  points,
}: {
  category: string;
  name: string;
  points: string[];
}) {
  return (
    <div className="rounded-xl border border-primary/30 bg-primary/5 px-3 py-2.5 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
          <Trophy className="size-3.5" />
          {category}
        </p>
        <p className="min-w-0 font-heading text-sm font-semibold">{name}</p>
      </div>
      {points.length > 0 && (
        <div className="mt-1.5 border-t border-primary/15 pt-1.5">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Why?
          </p>
          <ul className="mt-1 grid gap-x-6 gap-y-0.5 sm:grid-cols-2">
            {points.map((point, i) => (
              <li
                key={i}
                className="flex items-start gap-1.5 text-xs text-muted-foreground"
              >
                <span className="text-primary">✓</span>
                <span className="min-w-0 break-words">{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
