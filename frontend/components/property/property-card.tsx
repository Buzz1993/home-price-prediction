"use client";

// Reusable Property Card. Presentational only — it renders one ranked property
// (as returned by the backend search) and exposes a stage-to-tray toggle via
// props, so it can be reused by the chat search results, property search and
// saved properties pages. No business logic lives here.
//
// Every rich field (image, project name, locality/city, bed/bath/parking/
// balcony, area, cost per sqft, recommendation score) is rendered only when the
// backend provides it and hidden otherwise — the card never invents data.

import { useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Bath,
  BedDouble,
  Bookmark,
  Car,
  Fence,
  ImageOff,
  MapPin,
  Ruler,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import {
  formatArea,
  formatCr,
  formatPerSqft,
  splitList,
} from "@/features/dashboard/format";
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

// A backend value is renderable when it is neither missing nor empty.
function hasValue(value: number | string | null | undefined): boolean {
  return value !== null && value !== undefined && value !== "";
}

export function PropertyCard({
  property,
  staged = false,
  onToggleStage,
  saved = false,
  onToggleSave,
  savePending = false,
}: PropertyCardProps) {
  const [imageError, setImageError] = useState(false);

  const amenities = splitList(property.amenities_mcp);
  const primaryImage = property.image_urls?.[0];
  const showImage = Boolean(primaryImage) && !imageError;

  // Prefer the granular locality/city; fall back to the combined location.
  const place =
    [property.locality, property.city].filter(Boolean).join(", ") ||
    property.location;

  // Compact spec row — only the counts the backend actually returned.
  const specs = [
    { icon: BedDouble, label: "Beds", value: property.bed },
    { icon: Bath, label: "Baths", value: property.bath },
    { icon: Car, label: "Parking", value: property.parking },
    { icon: Fence, label: "Balcony", value: property.balcony },
  ].filter((spec) => hasValue(spec.value));

  return (
    <Card
      className={cn(
        "gap-0 overflow-hidden py-0 transition-shadow hover:shadow-md",
        staged && "border-primary ring-1 ring-primary"
      )}
    >
      {/* Primary image (image_urls[0]) with a consistent fallback placeholder. */}
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-muted">
        {showImage ? (
          // Plain img keeps arbitrary external listing URLs working without
          // per-domain Next image config; lazy + object-cover avoids distortion.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={primaryImage}
            alt={property.project_name || `Property ${property.id}`}
            loading="lazy"
            decoding="async"
            onError={() => setImageError(true)}
            className="size-full object-cover"
          />
        ) : (
          <div className="flex size-full items-center justify-center text-muted-foreground">
            <ImageOff className="size-8" />
          </div>
        )}

        {/* BHK badge overlaid on the image. */}
        <Badge className="absolute left-2 top-2 bg-primary text-primary-foreground shadow-sm">
          <BedDouble /> {property.bhk_type}
        </Badge>

        {/* Bookmark toggle — reuses the existing saved-property workflow. */}
        {onToggleSave && (
          <Button
            variant="secondary"
            size="icon-sm"
            aria-pressed={saved}
            aria-label={
              saved ? `Remove ${property.id} from saved` : `Save ${property.id}`
            }
            title={saved ? "Remove from saved" : "Save property"}
            disabled={savePending}
            onClick={() => onToggleSave(property.id)}
            className="absolute right-2 top-2 bg-background/80 backdrop-blur hover:bg-background"
          >
            <Bookmark className={cn(saved && "fill-primary text-primary")} />
          </Button>
        )}
      </div>

      <CardContent className="space-y-3 p-4">
        {/* Price and cost per sqft. */}
        <div className="flex items-end justify-between gap-2">
          <p className="text-lg font-semibold leading-none">
            {formatCr(property.price)}
          </p>
          {hasValue(property.cost_per_sqft) && (
            <span className="text-xs text-muted-foreground tabular-nums">
              {formatPerSqft(property.cost_per_sqft)}
            </span>
          )}
        </div>

        {/* Project name. */}
        {property.project_name && (
          <p className="truncate font-medium leading-tight">
            {property.project_name}
          </p>
        )}

        {/* Property ID — links to the details page. */}
        <Link
          href={`/property/${property.id}`}
          className="flex w-fit items-center gap-0.5 truncate font-mono text-xs text-muted-foreground hover:text-foreground hover:underline"
        >
          {property.id}
          <ArrowUpRight className="size-3 shrink-0" />
        </Link>

        {/* Locality / city. */}
        {place && (
          <p className="flex items-center gap-1 text-sm text-muted-foreground">
            <MapPin className="size-3.5 shrink-0" />
            <span className="truncate">{place}</span>
          </p>
        )}

        {/* Bed / bath / parking / balcony and area. */}
        {(specs.length > 0 || hasValue(property.area)) && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
            {specs.map((spec) => (
              <span key={spec.label} className="flex items-center gap-1">
                <spec.icon className="size-3.5 shrink-0" />
                <span className="tabular-nums">{spec.value}</span>
                <span>{spec.label}</span>
              </span>
            ))}
            {hasValue(property.area) && (
              <span className="flex items-center gap-1">
                <Ruler className="size-3.5 shrink-0" />
                <span className="tabular-nums">{formatArea(property.area)}</span>
              </span>
            )}
          </div>
        )}

        {/* Recommendation score (backend hybrid score). */}
        {property.search_score !== undefined && (
          <Badge variant="success" title="Recommendation score">
            <TrendingUp /> Recommendation {property.search_score.toFixed(2)}
          </Badge>
        )}

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
        <Button asChild variant="outline" size="sm">
          <Link href={`/property/${property.id}`}>
            Read More <ArrowUpRight />
          </Link>
        </Button>
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
