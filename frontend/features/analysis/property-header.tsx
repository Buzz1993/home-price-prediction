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

import { Home, type LucideIcon } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";
import { formatArea } from "@/features/dashboard/format";
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
    <div className="flex size-12 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-primary/10 sm:size-14">
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
    <header className="flex items-center gap-3 overflow-hidden rounded-xl border bg-card p-3 shadow-sm">
      <Thumbnail property={property} />

      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
          {Icon && <Icon className="size-3.5 shrink-0" />}
          {analysisLabel} · Property {index + 1}
        </p>
        <h3 className="truncate font-heading text-base font-semibold leading-tight">
          {title}
        </h3>
        <p className="truncate text-xs text-muted-foreground">
          {[location, ...facts].filter(Boolean).join(" • ") || (
            // Nothing beyond the id is known — keep it small, not prominent.
            <span className="font-mono">{id}</span>
          )}
        </p>
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1">
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
