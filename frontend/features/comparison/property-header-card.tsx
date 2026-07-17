"use client";

// Premium property header card (Phase 17.1 Goal 3, quick actions + hover
// polish in Phase 17.2) — the column header of the Property Overview
// comparison matrix. A compact mini-card: property image (slight zoom on
// hover), name, id, price, configuration, area, location, the backend advisor
// verdict (pill + star restatement), a gently pulsing "🏆 Overall Winner"
// ribbon, and quick actions that REUSE existing pages: property details,
// AI Analysis, Reports (the compared properties are already staged in the
// shared tray those pages read) and remove-from-comparison. Presentation
// only — every field is the verbatim backend value the matrix already
// receives.

import Link from "next/link";
import { Eye, FileText, Home, LineChart, MapPin, Trophy, X } from "lucide-react";
import { useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import { StarRow, ratingStars } from "@/features/analysis/ui/decision-summary";
import { formatArea, formatCr } from "@/features/dashboard/format";
import { cn } from "@/lib/utils";
import { hasText } from "./compare-utils";

// One small labelled icon action in the header card's action row.
function QuickAction({
  label,
  href,
  onClick,
  children,
}: {
  label: string;
  href?: string;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  const className =
    "flex size-7 items-center justify-center rounded-lg border bg-card text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {href ? (
          <Link href={href} aria-label={label} className={className}>
            {children}
          </Link>
        ) : (
          <button
            type="button"
            aria-label={label}
            onClick={onClick}
            className={className}
          >
            {children}
          </button>
        )}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

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
  onRemove,
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
  // Removes this property from the running comparison.
  onRemove?: (id: string) => void;
}) {
  const [imageError, setImageError] = useState(false);
  const showImage = Boolean(imageUrl) && !imageError;
  const stars = hasText(verdict) ? ratingStars(String(verdict)) : null;

  return (
    <div
      className={cn(
        "group w-48 space-y-2 rounded-xl border bg-card p-2 text-left shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg",
        // Overall winner: soft emerald border + subtle glow (Phase 18.6).
        isOverallWinner &&
          "border-emerald-400/50 bg-emerald-50/40 shadow-[0_0_0_1px_rgba(16,185,129,0.15),0_4px_16px_-4px_rgba(16,185,129,0.25)]"
      )}
    >
      <div className="relative flex h-24 items-center justify-center overflow-hidden rounded-lg bg-gradient-to-br from-primary/10 via-accent to-transparent">
        {showImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imageUrl}
            alt={name}
            loading="lazy"
            decoding="async"
            onError={() => setImageError(true)}
            className="size-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
        ) : (
          <Home className="size-6 text-primary/70" />
        )}
        {isOverallWinner && (
          <span className="absolute left-1.5 top-1.5 inline-flex items-center gap-1 rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-semibold text-white shadow-sm motion-safe:animate-pulse">
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
        <div className="flex flex-wrap items-center gap-1.5 px-0.5">
          {typeof stars === "number" && <StarRow count={stars} />}
          {hasText(verdict) && <StatusPill value={verdict as string | number} />}
        </div>
      )}

      {/* Quick actions — existing pages only (Phase 17.2, Goal 10). */}
      <div className="flex items-center gap-1 border-t px-0.5 pt-1.5">
        <QuickAction label="Open property details" href={`/property/${id}`}>
          <Eye className="size-3.5" />
        </QuickAction>
        <QuickAction label="Run individual analysis" href="/analysis">
          <LineChart className="size-3.5" />
        </QuickAction>
        <QuickAction label="Generate report" href="/reports">
          <FileText className="size-3.5" />
        </QuickAction>
        {onRemove && (
          <QuickAction
            label="Remove from comparison"
            onClick={() => onRemove(id)}
          >
            <X className="size-3.5" />
          </QuickAction>
        )}
      </div>
    </div>
  );
}
