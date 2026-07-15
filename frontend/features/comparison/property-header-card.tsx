"use client";

// Premium property header card (Phase 17.1, Goal 3) — the column header of
// the Property Overview comparison matrix. A compact mini-card: property
// image, name, id, price, configuration, area, location, the backend advisor
// verdict (pill + star restatement) and a "🏆 Overall Winner" badge when this
// property is the backend comparison winner. Presentation only — every field
// is the verbatim backend value the matrix already receives.

import { Home, MapPin, Trophy } from "lucide-react";
import { useState } from "react";

import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import { StarRow, ratingStars } from "@/features/analysis/ui/decision-summary";
import { formatArea, formatCr } from "@/features/dashboard/format";
import { cn } from "@/lib/utils";
import { hasText } from "./compare-utils";

export function PropertyHeaderCard({
  id,
  name,
  imageUrl,
  price,
  configuration,
  area,
  location,
  verdict,
  isOverallWinner,
}: {
  id: string;
  name: string;
  imageUrl?: string;
  price?: unknown;
  configuration?: unknown;
  area?: unknown;
  location?: unknown;
  // The backend advisor verdict for this property (pill + stars).
  verdict?: unknown;
  isOverallWinner: boolean;
}) {
  const [imageError, setImageError] = useState(false);
  const showImage = Boolean(imageUrl) && !imageError;
  const stars = hasText(verdict) ? ratingStars(String(verdict)) : null;

  return (
    <div
      className={cn(
        "w-48 space-y-2 rounded-xl border bg-card p-2 text-left shadow-sm transition-shadow hover:shadow-md",
        isOverallWinner && "border-primary/40 bg-primary/5"
      )}
    >
      <div className="relative flex h-24 items-center justify-center overflow-hidden rounded-lg bg-primary/10">
        {showImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imageUrl}
            alt={name}
            loading="lazy"
            decoding="async"
            onError={() => setImageError(true)}
            className="size-full object-cover"
          />
        ) : (
          <Home className="size-6 text-primary" />
        )}
        {isOverallWinner && (
          <span className="absolute left-1.5 top-1.5 inline-flex items-center gap-1 rounded-full bg-primary px-2 py-0.5 text-[10px] font-semibold text-primary-foreground shadow-sm">
            <Trophy className="size-3" />
            Overall Winner
          </span>
        )}
      </div>

      <div className="min-w-0 space-y-0.5 px-0.5">
        <p className="truncate font-heading text-sm font-semibold text-foreground">
          {name}
        </p>
        <p className="truncate font-mono text-[10px] font-normal text-muted-foreground">
          {id}
        </p>
      </div>

      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 px-0.5">
        {hasText(price) && (
          <span className="font-heading text-sm font-bold tabular-nums text-primary">
            {formatCr(price as number | string)}
          </span>
        )}
        {hasText(configuration) && (
          <span className="text-[11px] font-medium normal-case text-muted-foreground">
            {String(configuration)}
          </span>
        )}
        {hasText(area) && (
          <span className="text-[11px] font-normal normal-case text-muted-foreground">
            {formatArea(area as number | string)}
          </span>
        )}
      </div>

      {hasText(location) && (
        <p className="flex items-center gap-1 px-0.5 text-[11px] font-normal normal-case text-muted-foreground">
          <MapPin className="size-3 shrink-0" />
          <span className="min-w-0 truncate">{String(location)}</span>
        </p>
      )}

      {(hasText(verdict) || typeof stars === "number") && (
        <div className="flex flex-wrap items-center gap-1.5 px-0.5 pb-0.5">
          {typeof stars === "number" && <StarRow count={stars} />}
          {hasText(verdict) && <StatusPill value={verdict as string | number} />}
        </div>
      )}
    </div>
  );
}
