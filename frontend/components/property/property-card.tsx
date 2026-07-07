"use client";

// Reusable Property Card. Presentational only — it renders one ranked property
// (as returned by the backend search) and exposes a stage-to-tray toggle via
// props, so it can be reused by the chat search results, property search and
// saved properties pages. No business logic lives here.

import Link from "next/link";
import { ArrowUpRight, BedDouble, Bookmark, MapPin, Sparkles } from "lucide-react";

import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { formatCr, splitList } from "@/features/dashboard/format";
import type { PropertyCardData } from "@/types/dashboard";

type PropertyCardProps = {
  property: PropertyCardData;
  // Whether the property is staged in the evaluation tray.
  staged?: boolean;
  // Called with the property id when the user toggles staging.
  onToggleStage?: (id: string) => void;
  // Whether the property is in the user's saved list.
  saved?: boolean;
  // Called with the property id when the user toggles the saved state.
  onToggleSave?: (id: string) => void;
  // Disables the save toggle while a save/remove request is in flight.
  savePending?: boolean;
};

export function PropertyCard({
  property,
  staged = false,
  onToggleStage,
  saved = false,
  onToggleSave,
  savePending = false,
}: PropertyCardProps) {
  const amenities = splitList(property.amenities_mcp);

  return (
    <Card
      className={cn(
        "gap-0 py-0 transition-colors",
        staged && "border-primary ring-1 ring-primary"
      )}
    >
      <CardContent className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 space-y-1">
            <p className="text-lg font-semibold leading-none">
              {formatCr(property.price)}
            </p>
            <Link
              href={`/property/${property.id}`}
              className="flex items-center gap-0.5 truncate font-mono text-xs text-muted-foreground hover:text-foreground hover:underline"
            >
              {property.id}
              <ArrowUpRight className="size-3 shrink-0" />
            </Link>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {onToggleSave && (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-pressed={saved}
                aria-label={saved ? `Remove ${property.id} from saved` : `Save ${property.id}`}
                title={saved ? "Remove from saved" : "Save property"}
                disabled={savePending}
                onClick={() => onToggleSave(property.id)}
              >
                <Bookmark className={cn(saved && "fill-primary text-primary")} />
              </Button>
            )}
            <Badge variant="secondary">
              <BedDouble /> {property.bhk_type}
            </Badge>
          </div>
        </div>

        <p className="flex items-center gap-1 text-sm text-muted-foreground">
          <MapPin className="size-3.5 shrink-0" />
          <span className="truncate">{property.location}</span>
        </p>

        {amenities.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {amenities.slice(0, 4).map((a) => (
              <Badge key={a} variant="outline">
                {a}
              </Badge>
            ))}
            {amenities.length > 4 && (
              <Badge variant="outline">+{amenities.length - 4}</Badge>
            )}
          </div>
        )}

        {property.why_recommended && (
          <p className="flex gap-1.5 text-xs text-muted-foreground">
            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-primary" />
            <span className="line-clamp-3">{property.why_recommended}</span>
          </p>
        )}
      </CardContent>

      <CardFooter className="justify-between border-t p-4">
        {property.search_score !== undefined ? (
          <span className="text-xs text-muted-foreground tabular-nums">
            BM25 {property.search_score.toFixed(4)}
          </span>
        ) : (
          <span />
        )}
        {onToggleStage && (
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Checkbox
              checked={staged}
              onCheckedChange={() => onToggleStage(property.id)}
              aria-label={`Stage property ${property.id}`}
            />
            {staged ? "Staged" : "Stage to tray"}
          </label>
        )}
      </CardFooter>
    </Card>
  );
}
