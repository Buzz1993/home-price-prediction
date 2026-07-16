"use client";

// Side-by-side property comparison matrix (Phase 17.0). Renders the full
// backend property records (GET /property/{id}) row-wise: the known fields get
// dedicated formatting and smart highlighting, and EVERY remaining backend
// field is rendered under Additional Information — a backend field is never
// silently dropped (same rule as the Property Details page). Highlighting only
// compares existing backend values (compare-utils); ratings / scores read
// higher-is-better and distances lower-is-better, everything else stays
// neutral.

import { Building2, ListPlus } from "lucide-react";

import {
  formatArea,
  formatCell,
  formatCr,
  formatPerSqft,
  humanizeKey,
  splitList,
} from "@/features/dashboard/format";
import { formatValue } from "@/features/property/property-fields";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import type { PropertyDetail, SearchResult } from "@/types/dashboard";
import {
  allIdentical,
  barFractions,
  hasText,
  rankNumericCells,
  relativeNotes,
  type CellTone,
  type Direction,
} from "./compare-utils";
import {
  MatrixTable,
  type MatrixCell,
  type MatrixColumn,
  type MatrixRowData,
} from "./matrix-table";
import { PropertyHeaderCard } from "./property-header-card";

// Fields the dedicated rows below already render; everything else the backend
// returns flows into the Additional Information table.
const CONSUMED_KEYS = new Set([
  "id",
  "project_name",
  "builder",
  "location",
  "locality",
  "city",
  "price",
  "area",
  "costpersqft",
  "bhk_type",
  "bed",
  "bath",
  "balcony",
  "parking",
  "property_type",
  "status",
  "amenities_mcp",
  "features_mcp",
  "image_urls",
  "ap_pjt_url",
]);

// One property per column: the backend record first, the workspace search row
// as fallback so the matrix still renders while a record failed to load.
type MatrixSource = {
  detail: PropertyDetail | null;
  fallback: SearchResult | undefined;
};

function fieldOf(source: MatrixSource, key: string): unknown {
  const fromDetail = source.detail?.[key];
  if (hasText(fromDetail)) return fromDetail;
  return (source.fallback as Record<string, unknown> | undefined)?.[key];
}

// Build a highlighted row from raw backend values. `direction` opts the row
// into smart highlighting AND the Goal-2 proportional value bars; `note`
// additionally adds the Goal-6 relative-difference line to the winning cell.
function buildRow(
  label: string,
  values: unknown[],
  {
    tooltip,
    direction,
    badge,
    note,
    names,
    render,
  }: {
    tooltip?: string;
    direction?: Direction;
    badge?: string;
    note?: (diff: number, otherName: string) => string;
    names?: string[];
    render?: (value: unknown) => React.ReactNode;
  } = {}
): MatrixRowData | null {
  if (!values.some(hasText)) return null;

  const tones: CellTone[] = direction
    ? rankNumericCells(values, direction)
    : values.map(() => "neutral");
  const bars = direction ? barFractions(values, direction) : null;
  const notes =
    direction && note && names
      ? relativeNotes(values, direction, names, note)
      : null;

  const cells: MatrixCell[] = values.map((value, i) => ({
    // formatCell cleans raw float noise from open backend values; renderers
    // override it for typed cells.
    content: hasText(value)
      ? render
        ? render(value)
        : formatCell(value as string | number)
      : "—",
    tone: tones[i],
    badge,
    barFraction: bars?.[i] ?? null,
    note: notes?.[i] ?? null,
  }));

  return { label, tooltip, identical: allIdentical(values), cells };
}

export function PropertyMatrix({
  columns,
  sources,
  hideIdentical = false,
  onRemove,
}: {
  columns: MatrixColumn[];
  sources: MatrixSource[];
  // "Show Only Differences" (Phase 17.2) — forwarded to both tables.
  hideIdentical?: boolean;
  // Removes a property from the running comparison (header quick action).
  onRemove?: (id: string) => void;
}) {
  const values = (key: string) => sources.map((source) => fieldOf(source, key));
  // Column names for the Goal-6 relative-difference notes.
  const names = columns.map((column) => column.name);

  // Premium header cards (Goal 3): the property image moves into the header,
  // so the matrix no longer needs a separate Property Image row.
  const headerColumns: MatrixColumn[] = columns.map((column, i) => ({
    ...column,
    header: (
      <PropertyHeaderCard
        id={column.id}
        name={column.name}
        imageUrl={
          sources[i]?.detail?.image_urls?.[0] ??
          sources[i]?.fallback?.image_urls?.[0]
        }
        price={fieldOf(sources[i], "price")}
        configuration={fieldOf(sources[i], "bhk_type")}
        area={fieldOf(sources[i], "area")}
        location={fieldOf(sources[i], "location")}
        verdict={column.verdict}
        isOverallWinner={Boolean(column.isWinner)}
        onRemove={onRemove}
      />
    ),
  }));

  // --- Known fields, with formatting + smart highlighting ------------------
  // The property image and name now live in the premium header card, so the
  // matrix body starts at the identifying fields.
  const knownRows = [
    buildRow("Property Name", values("project_name")),
    buildRow("Property ID", values("id"), {
      render: (v) => <span className="font-mono text-xs">{String(v)}</span>,
    }),
    buildRow("Location", values("location")),
    buildRow("City", values("city")),
    buildRow("Builder", values("builder")),
    buildRow("Price", values("price"), {
      tooltip: "The asking price of the property, in ₹ Crores.",
      direction: "lowest",
      badge: "Lowest price",
      names,
      note: (diff, other) => `${formatCr(diff)} cheaper than ${other}`,
      render: (v) => formatCr(v as number | string),
    }),
    buildRow("Area", values("area"), {
      tooltip: "The property area in square feet.",
      direction: "highest",
      badge: "Largest",
      names,
      note: (diff, other) =>
        `${Math.round(diff).toLocaleString("en-IN")} sqft larger than ${other}`,
      render: (v) => formatArea(v as number | string),
    }),
    buildRow("Price per sqft", values("costpersqft"), {
      tooltip: "Property price divided by area — lower means more space per rupee.",
      direction: "lowest",
      badge: "Lowest ₹/sqft",
      names,
      note: (diff, other) =>
        `₹${Math.round(diff).toLocaleString("en-IN")}/sqft less than ${other}`,
      render: (v) => formatPerSqft(v as number | string),
    }),
    buildRow("Configuration", values("bhk_type")),
    buildRow("Bedrooms", values("bed"), { direction: "highest" }),
    buildRow("Bathrooms", values("bath"), { direction: "highest" }),
    buildRow("Balconies", values("balcony"), { direction: "highest" }),
    buildRow("Parking", values("parking"), { direction: "highest" }),
    buildRow("Property Type", values("property_type")),
    buildRow("Status", values("status"), {
      render: (v) => <StatusPill value={v as string | number} />,
    }),
    buildRow(
      "Amenities Count",
      sources.map((source) => {
        const amenities = fieldOf(source, "amenities_mcp");
        const count = splitList(
          hasText(amenities) ? String(amenities) : null
        ).length;
        return count > 0 ? count : null;
      }),
      {
        tooltip:
          "Number of amenities listed for the property (pool, gym, security, …).",
        direction: "highest",
        badge: "Most amenities",
        names,
        note: (diff, other) =>
          `${Math.round(diff)} more amenities than ${other}`,
      }
    ),
    buildRow(
      "Features Count",
      sources.map((source) => {
        const features = fieldOf(source, "features_mcp");
        const count = splitList(
          hasText(features) ? String(features) : null
        ).length;
        return count > 0 ? count : null;
      }),
      {
        tooltip: "Number of listed unit features (flooring, fittings, …).",
        direction: "highest",
      }
    ),
    buildRow("Original Listing", values("ap_pjt_url"), {
      render: (v) => (
        <a
          href={String(v)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs font-medium text-primary underline-offset-2 hover:underline"
        >
          View listing ↗
        </a>
      ),
    }),
  ].filter((row): row is MatrixRowData => row !== null);

  // --- Additional Information: every other backend field -------------------
  // Union of the remaining keys across all records, so a field present on only
  // one property still gets a row ("—" for the others). Ratings / scores are
  // higher-is-better and distances lower-is-better; anything else is neutral.
  const extraKeys = Array.from(
    new Set(
      sources.flatMap((source) =>
        Object.entries(source.detail ?? {})
          .filter(([key, value]) => !CONSUMED_KEYS.has(key) && hasText(value))
          .map(([key]) => key)
      )
    )
  ).sort();

  const extraRows = extraKeys
    .map((key) => {
      const lower = key.toLowerCase();
      const direction: Direction | undefined =
        lower.includes("rating") || lower.includes("score")
          ? "highest"
          : lower.includes("distance")
            ? "lowest"
            : undefined;
      return buildRow(
        humanizeKey(key),
        sources.map((source) => source.detail?.[key]),
        {
          direction,
          badge: direction ? "Best" : undefined,
          render: (v) => formatValue(v),
        }
      );
    })
    .filter((row): row is MatrixRowData => row !== null);

  return (
    <div className="space-y-3">
      <MatrixTable
        title="Property Overview"
        icon={Building2}
        description="Every field returned by the backend, compared side-by-side. The best value in each row is highlighted."
        columns={headerColumns}
        rows={knownRows}
        hideIdentical={hideIdentical}
      />
      {extraRows.length > 0 && (
        <MatrixTable
          title="Additional Information"
          icon={ListPlus}
          columns={columns}
          rows={extraRows}
          hideIdentical={hideIdentical}
        />
      )}
    </div>
  );
}
