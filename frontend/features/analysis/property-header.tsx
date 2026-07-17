"use client";

// CompactPropertyHeader (Phase 15.18). A 70–90px property strip that replaces
// the tall green banner in every analysis renderer: thumbnail (when the
// backend already returned image_urls for this property), property name,
// location, configuration chips and an optional verbatim status pill.
//
// Purely presentational: it only LOOKS UP the property in the workspace's
// accumulated search-result collection (the same backend rows the tray and map
// render from) so analyses can show the real name/location/config instead of a
// bare card id. Nothing is fetched or computed — if the property isn't in the
// collection, it gracefully falls back to whatever the analysis row carried.

import { Home, MapPin, type LucideIcon } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";
import { formatArea, formatCr, formatScore } from "@/features/dashboard/format";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import type { SearchResult } from "@/types/dashboard";
import { StatusPill, hasValue } from "./ui/analysis-ui";

// Small square thumbnail from the backend image_urls, with an elegant
// placeholder fallback (same plain-<img> approach as PropertyCard).
function Thumbnail({ property }: { property?: SearchResult }) {
  const [imageError, setImageError] = useState(false);
  const src = property?.image_urls?.[0];
  const showImage = Boolean(src) && !imageError;

  return (
    <div className="flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-primary/10 sm:size-16">
      {showImage ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={property?.project_name || "Property"}
          loading="lazy"
          decoding="async"
          onError={() => setImageError(true)}
          className="size-full object-cover"
        />
      ) : (
        <Home className="size-5 text-primary" />
      )}
    </div>
  );
}

export function CompactPropertyHeader({
  id,
  index,
  analysisLabel,
  icon: Icon,
  // Fallbacks straight from the analysis row (rendered only when the property
  // is not in the workspace collection, e.g. valuation rows carry project_name
  // and prediction rows carry location).
  fallbackTitle,
  fallbackSubtitle,
  status,
}: {
  id: string;
  index: number;
  analysisLabel: string;
  icon?: LucideIcon;
  fallbackTitle?: string;
  fallbackSubtitle?: string;
  status?: string | number;
}) {
  const { properties } = useWorkspace();
  const property = properties.find((p) => p.id === id);

  // Prefer the human-readable name from the backend search row; fall back to
  // whatever the analysis row itself provided, then to the raw id.
  const title = property?.project_name || fallbackTitle || id;
  const location =
    property?.location ||
    [property?.locality, property?.city].filter(Boolean).join(", ") ||
    fallbackSubtitle;

  // Configuration facts, shown verbatim, only when the backend provided them.
  const facts = property
    ? [
        property.bhk_type,
        hasValue(property.area) ? formatArea(property.area) : null,
      ].filter(Boolean)
    : [];

  return (
    <header className="flex items-center gap-3.5 overflow-hidden rounded-xl border bg-card p-3.5 shadow-float">
      <Thumbnail property={property} />

      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
          {Icon && (
            <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-primary/10">
              <Icon className="size-3.5" />
            </span>
          )}
          {analysisLabel} · Property {index + 1}
        </p>
        <h3 className="mt-0.5 truncate font-heading text-base font-semibold leading-tight sm:text-lg">
          {title}
        </h3>
        <p className="mt-0.5 flex items-center gap-1 truncate text-[13px] text-muted-foreground">
          {location && <MapPin className="size-3 shrink-0" />}
          <span className="truncate">
            {[location, ...facts].filter(Boolean).join(" • ") || (
              // Nothing beyond the id is known — keep it small, not prominent.
              <span className="font-mono">{id}</span>
            )}
          </span>
        </p>
        {/* Price + AI score quick facts, straight from the backend search row. */}
        {property && (
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[13px]">
            {hasValue(property.price) && (
              <span className="font-semibold tabular-nums">
                {formatCr(property.price)}
              </span>
            )}
            {property.search_score !== undefined && (
              <span className="font-medium text-primary">
                AI {formatScore(property.search_score)}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1.5">
        {hasValue(status) && <StatusPill value={status} />}
        {/* The card id stays available as small metadata when a real name is
            shown instead. */}
        {title !== id && (
          <span
            className={cn(
              "max-w-28 truncate font-mono text-[10px] text-muted-foreground/70"
            )}
            title={id}
          >
            {id}
          </span>
        )}
      </div>
    </header>
  );
}
