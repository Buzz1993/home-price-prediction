"use client";

// Reusable Property Card. Presentational only — it renders one ranked property
// (as returned by the backend search) and exposes a stage-to-tray toggle via
// props, so it can be reused by the chat search results, property search and
// saved properties pages. No business logic lives here.
//
// Every rich field (image, project name, locality/city, bed/bath/parking/
// balcony, area, cost per sqft, recommendation score) is rendered only when the
// backend provides it and hidden otherwise — the card never invents data.
//
// Phase 18.3 premium polish: taller imagery with a readability gradient and a
// premium lavender "Image Unavailable" placeholder, a dominant price line, a
// larger purple project name, lavender amenity pills, a gradient AI badge, a
// gradient View Listing CTA with an outlined Read More beside it, and more
// generous vertical rhythm. Presentation only — data, props and behavior are
// unchanged.

import { useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Bath,
  BedDouble,
  Bookmark,
  Building2,
  Car,
  Fence,
  MapPin,
  Ruler,
  Sparkles,
} from "lucide-react";

import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { OriginalListingButton } from "@/components/property/original-listing-button";
import { cn } from "@/lib/utils";
import {
  formatArea,
  formatCr,
  formatPerSqft,
  formatScore,
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
  // Opt-in horizontal layout (image left, details right) from the md breakpoint
  // up — used by the workspace results list to fit more properties on screen.
  // Below md the card always falls back to the default vertical layout. No
  // information changes; only the arrangement does.
  horizontal?: boolean;
  // Opt-in compact density (Phase 18.9, tightened in 18.11 and again ~20% in
  // 18.12 for Saved Properties): a much shorter image, tighter padding and
  // rhythm, smaller price/name/badges. Every field still renders — only the
  // spacing scale changes.
  compact?: boolean;
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
  horizontal = false,
  compact = false,
}: PropertyCardProps) {
  const [imageError, setImageError] = useState(false);

  const amenities = splitList(property.amenities_mcp);
  const primaryImage = property.image_urls?.[0];
  const showImage = Boolean(primaryImage) && !imageError;
  const imageCount = property.image_urls?.length ?? 0;

  // Prefer the granular locality/city; fall back to the combined location.
  const place =
    [property.locality, property.city].filter(Boolean).join(", ") ||
    property.location;

  // Compact spec row — only the counts the backend actually returned.
  const specs = [
    { icon: BedDouble, label: "Beds", value: property.bed },
    { icon: Bath, label: "Baths", value: property.bath },
    { icon: Fence, label: "Balconies", value: property.balcony },
    { icon: Car, label: "Parking", value: property.parking },
  ].filter((spec) => hasValue(spec.value));

  return (
    <Card
      className={cn(
        // Premium floating listing card (Phase 18.3): slightly stronger lift +
        // deeper soft shadow on hover; `group` drives the image hover-zoom
        // below. 150–200ms transitions keep interactions crisp.
        "group gap-0 overflow-hidden py-0 transition-all duration-200 ease-out hover:-translate-y-1 hover:shadow-float-lg",
        // Horizontal layout (md+): image on the left, details on the right.
        horizontal && "md:flex-row",
        // Staged: the ENTIRE card switches to a rich (but still elegant)
        // premium light-purple appearance — a clear purple gradient wash, a
        // deeper purple border, a purple ring and a soft glow — so staged
        // properties are immediately recognizable. Kept light enough that all
        // text stays fully readable. Reverts smoothly when unstaged.
        staged &&
          "border-primary/60 bg-gradient-to-br from-primary/20 via-primary/10 to-primary/5 shadow-lg shadow-primary/25 ring-2 ring-primary/40"
      )}
    >
      {/* Primary image (image_urls[0]) with a premium lavender placeholder
          fallback. Vertical: ~25% taller than the previous 4/3 frame (3/2 →
          16/13 ≈ +23–25% height for the same width). Horizontal: a fixed-width
          column on the left that stretches to the card's height. */}
      <div
        className={cn(
          "relative aspect-[16/13] w-full overflow-hidden",
          compact && "aspect-[5/2]",
          horizontal &&
            "md:aspect-auto md:w-56 md:shrink-0 md:self-stretch lg:w-72"
        )}
      >
        {showImage ? (
          <>
            {/* Plain img keeps arbitrary external listing URLs working without
                per-domain Next image config; lazy + object-cover avoids
                distortion. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={primaryImage}
              alt={property.project_name || `Property ${property.id}`}
              loading="lazy"
              decoding="async"
              onError={() => setImageError(true)}
              className="size-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.05]"
            />
            {/* Subtle dark gradient at the bottom so the overlaid badges stay
                readable on bright photos. Purely decorative. */}
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/35 to-transparent"
            />
          </>
        ) : (
          // Premium placeholder — soft lavender gradient, building icon and an
          // explicit label so a missing image looks intentional, never broken.
          <div className="flex size-full flex-col items-center justify-center gap-2.5 bg-gradient-to-br from-secondary via-accent to-primary/10">
            <div className="flex size-12 items-center justify-center rounded-2xl bg-card/80 text-primary shadow-float ring-1 ring-primary/10">
              <Building2 className="size-6" />
            </div>
            <p className="text-xs font-medium text-muted-foreground">
              Image Unavailable
            </p>
          </div>
        )}

        {/* BHK badge overlaid on the image. */}
        <Badge
          className={cn(
            "bg-brand-gradient absolute left-2.5 top-2.5 border-transparent text-primary-foreground shadow-sm",
            compact && "left-2 top-2 px-1.5 py-0.5 text-[0.68rem]"
          )}
        >
          <BedDouble /> {property.bhk_type}
        </Badge>

        {/* Image counter — how many photos the listing carries. */}
        {imageCount > 1 && (
          <Badge
            className={cn(
              "absolute bottom-2.5 left-2.5 border-transparent bg-black/55 text-white backdrop-blur-sm",
              compact && "bottom-2 left-2 px-1.5 py-0.5 text-[0.68rem]"
            )}
          >
            {imageCount} photos
          </Badge>
        )}

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
            className="absolute right-2.5 top-2.5 bg-background/80 shadow-sm backdrop-blur transition-transform duration-200 hover:scale-105 hover:bg-background"
          >
            <Bookmark className={cn(saved && "fill-primary text-primary")} />
          </Button>
        )}
      </div>

      {/* Details column — vertical stack of every backend field. In horizontal
          mode it fills the space beside the image and stays a bounded flex
          column so long text truncates instead of pushing the image. */}
      <div
        className={cn(
          "flex min-w-0 flex-col",
          horizontal && "md:flex-1"
        )}
      >
      <CardContent className={cn("space-y-3.5 p-5", compact && "space-y-1.5 p-3")}>
        {/* Price — the dominant element. The property id sits top-right as a
            low-emphasis monospace reference (Phase 18.9 hierarchy: price >
            name > ₹/sq.ft > location > specs); it still links to details. */}
        <div className="flex items-start justify-between gap-2">
          <p
            className={cn(
              "font-heading text-2xl font-bold leading-none tracking-tight",
              compact && "text-base"
            )}
          >
            {formatCr(property.price)}
          </p>
          <Link
            href={`/property/${property.id}`}
            className="flex max-w-[45%] shrink-0 items-center gap-0.5 truncate font-mono text-[0.68rem] text-muted-foreground/70 transition-colors hover:text-foreground hover:underline"
            title={`Open property ${property.id}`}
          >
            <span className="truncate">{property.id}</span>
            <ArrowUpRight className="size-3 shrink-0" />
          </Link>
        </div>

        {/* Project name — bold purple, a clear step below the price. */}
        {property.project_name && (
          <p
            className={cn(
              "truncate pt-0.5 text-base font-bold leading-tight text-primary",
              compact && "pt-0 text-[0.8125rem]"
            )}
          >
            {property.project_name}
          </p>
        )}

        {/* Cost per sq.ft — small secondary line under the name, more visible
            than the card id but clearly quieter than the price. */}
        {hasValue(property.costpersqft) && (
          <p className="text-xs font-medium text-muted-foreground tabular-nums">
            {formatPerSqft(property.costpersqft)}
          </p>
        )}

        {/* Locality / city — muted, icon-aligned. */}
        {place && (
          <p
            className={cn(
              "flex items-center gap-1.5 text-sm text-muted-foreground",
              compact && "text-xs"
            )}
          >
            <MapPin
              className={cn("size-4 shrink-0 text-primary/60", compact && "size-3.5")}
            />
            <span className="truncate">{place}</span>
          </p>
        )}

        {/* Bed / bath / balcony / parking and area — consistent Lucide icons,
            equal spacing. */}
        {(specs.length > 0 || hasValue(property.area)) && (
          <div
            className={cn(
              "flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-muted-foreground",
              compact && "gap-x-3 gap-y-1 text-xs"
            )}
          >
            {specs.map((spec) => (
              <span key={spec.label} className="flex items-center gap-1.5">
                <spec.icon className={cn("size-4 shrink-0", compact && "size-3.5")} />
                <span className="font-medium tabular-nums text-foreground/80">
                  {spec.value}
                </span>
                <span>{spec.label}</span>
              </span>
            ))}
            {hasValue(property.area) && (
              <span className="flex items-center gap-1.5">
                <Ruler className={cn("size-4 shrink-0", compact && "size-3.5")} />
                <span className="font-medium tabular-nums text-foreground/80">
                  {formatArea(property.area)}
                </span>
              </span>
            )}
          </div>
        )}

        {/* Recommendation score (backend hybrid score) — premium purple
            gradient pill. Displays ONLY the backend value. */}
        {property.search_score !== undefined && (
          <Badge
            title="Recommendation score"
            className={cn(
              "bg-brand-gradient shadow-brand-glow w-fit border-transparent px-3 py-1 text-primary-foreground",
              compact && "px-2 py-0.5 text-[0.68rem]"
            )}
          >
            <Sparkles /> AI Recommended · Score{" "}
            {formatScore(property.search_score)}
          </Badge>
        )}

        {/* Amenity chips — lavender pills with a soft hover wash. */}
        {amenities.length > 0 && (
          <div className={cn("flex flex-wrap gap-2", compact && "gap-1.5")}>
            {amenities.slice(0, 4).map((a) => (
              <Badge
                key={a}
                variant="secondary"
                className={cn(
                  "px-3 py-1 transition-colors duration-150 hover:bg-accent hover:text-accent-foreground",
                  compact && "px-2 py-0.5 text-[0.68rem]"
                )}
              >
                {a}
              </Badge>
            ))}
            {amenities.length > 4 && (
              <Badge
                variant="secondary"
                className={cn(
                  "px-3 py-1 transition-colors duration-150 hover:bg-accent hover:text-accent-foreground",
                  compact && "px-2 py-0.5 text-[0.68rem]"
                )}
              >
                +{amenities.length - 4}
              </Badge>
            )}
          </div>
        )}

        {/* Why recommended — up to 3 relaxed lines, clamped (faded by clamp). */}
        {property.why_recommended && (
          <p className="flex gap-2 text-xs leading-relaxed text-muted-foreground">
            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-primary" />
            <span className="line-clamp-3">{property.why_recommended}</span>
          </p>
        )}
      </CardContent>

      {/* Actions — primary gradient View Listing + outlined purple Read More,
          consistent heights; staging keeps its existing checkbox workflow. */}
      <CardFooter
        className={cn(
          "mt-auto flex-wrap gap-3 border-t p-5 pt-4",
          compact && "gap-1.5 p-3 pt-2"
        )}
      >
        <div
          className={cn(
            "flex flex-1 flex-wrap items-center gap-2.5",
            compact && "gap-1.5"
          )}
        >
          {/* Original listing — reuses the shared action; hidden when no valid
              backend URL exists. Primary CTA styling comes from the default
              button variant (purple gradient, hover lift, soft glow). */}
          <OriginalListingButton
            url={property.ap_pjt_url}
            label="View Listing"
            variant="default"
            size="sm"
          />
          <Button
            asChild
            variant="outline"
            size="sm"
            className="border-primary/30 bg-card text-primary hover:border-primary/50 hover:bg-accent hover:text-accent-foreground"
          >
            <Link href={`/property/${property.id}`}>
              Read More <ArrowUpRight />
            </Link>
          </Button>
        </div>
        {onToggleStage && (
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Checkbox
              checked={staged}
              onCheckedChange={() => onToggleStage(property.id)}
              aria-label={`Stage property ${property.id}`}
            />
            {staged ? (
              // Staged badge — soft green rounded pill.
              <Badge variant="success" className="px-3 py-1">
                Staged
              </Badge>
            ) : (
              "Stage to tray"
            )}
          </label>
        )}
      </CardFooter>
      </div>
    </Card>
  );
}
