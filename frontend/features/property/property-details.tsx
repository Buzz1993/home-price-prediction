"use client";

// Rich Property Details view (Phase 14.4).
//
// Renders every field returned by GET /property/{id}. The backend is the only
// source of truth: the page never assumes a fixed schema, never invents data,
// and automatically categorizes each backend field into a logical section
// (project_docs/04_UI.md). Any field that matches no known section is rendered
// under "Additional Information", so the page stays compatible with future
// backend fields without code changes.
//
// Reused components: PropertyImageGallery (Phase 14.2), OriginalListingButton
// (Phase 14.3), PropertyLocationMap placeholder, the shared UI primitives, the
// saved-property workflow, and the format helpers. No business logic lives here.

import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowUpDown,
  Bath,
  Baby,
  BedDouble,
  Bookmark,
  Building2,
  Car,
  CheckCircle2,
  Dumbbell,
  Fence,
  FileSearch,
  Flame,
  MapPin,
  Ruler,
  Shield,
  Sofa,
  Sparkles,
  Star,
  Trees,
  Users,
  Waves,
  Wifi,
  Wind,
  type LucideIcon,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { PropertyImageGallery } from "@/components/property/property-image-gallery";
import { OriginalListingButton } from "@/components/property/original-listing-button";
import { PropertyLocationMap } from "@/components/property/property-location-map";
import {
  InteractivePropertyMap,
  getMapCoords,
  type MapProperty,
} from "@/components/property/interactive-property-map";
import {
  formatArea,
  formatCr,
  humanizeKey,
  splitList,
} from "@/features/dashboard/format";
import {
  useRemoveSavedProperty,
  useSaveProperty,
  useSavedProperties,
} from "@/features/saved/use-saved";
import { cn } from "@/lib/utils";
import { ApiError } from "@/lib/api-client";
import type { NotFoundResponse, PropertyDetail } from "@/types/dashboard";
import {
  categorizeFields,
  formatValue,
  isEmptyValue,
  toList,
  type FieldEntry,
} from "./property-fields";
import { usePropertyDetails } from "./use-property-details";

// Narrow the backend response, which is a property record or `{ error }`.
function isNotFound(
  data: PropertyDetail | NotFoundResponse
): data is NotFoundResponse {
  return "error" in data;
}

function BackLink() {
  return (
    <Button asChild variant="ghost" size="sm" className="-ml-2 w-fit">
      <Link href="/chat">
        <ArrowLeft /> Back to Copilot
      </Link>
    </Button>
  );
}

export function PropertyDetails({ id }: { id: string }) {
  const { data, error, isLoading, isError, isFetching, refetch } =
    usePropertyDetails(id);

  // The backend returns HTTP 404 when no property matches the id.
  const notFound = error instanceof ApiError && error.status === 404;

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <BackLink />
        <Skeleton className="aspect-[4/3] w-full rounded-xl sm:aspect-[16/9]" />
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-40 rounded-xl" />
      </div>
    );
  }

  if (isError && !notFound) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <BackLink />
        <ErrorState
          title="Could not load this property"
          description="Something went wrong while loading the property details. Please try again."
          onRetry={() => refetch()}
          retrying={isFetching}
        />
      </div>
    );
  }

  if (notFound || !data || isNotFound(data)) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <BackLink />
        <Card>
          <CardContent className="p-6">
            <EmptyState
              icon={FileSearch}
              title="Property not found"
              description={`No property matches id ${id}.`}
            />
          </CardContent>
        </Card>
      </div>
    );
  }

  return <PropertyDetailsContent property={data} />;
}

// ---------------------------------------------------------------------------
// Quick Highlights — a candidate list of common spec keys, each rendered only
// when the backend returns it. The matched key is marked consumed so it is not
// repeated elsewhere on the page.
const HIGHLIGHT_SPECS: { keys: string[]; label: string; icon: LucideIcon }[] = [
  { keys: ["bed", "beds", "bedrooms", "bedroom"], label: "Bedrooms", icon: BedDouble },
  { keys: ["bath", "baths", "bathrooms", "bathroom"], label: "Bathrooms", icon: Bath },
  { keys: ["parking", "car_parking", "reserved_parking"], label: "Parking", icon: Car },
  { keys: ["balcony", "balconies"], label: "Balcony", icon: Fence },
  { keys: ["furnish", "furnishing", "furnished"], label: "Furnishing", icon: Sofa },
  { keys: ["area", "super_built_up_area", "built_up_area", "carpet_area"], label: "Area", icon: Ruler },
  { keys: ["floor"], label: "Floor", icon: Building2 },
];

function PropertyDetailsContent({ property }: { property: PropertyDetail }) {
  const record = property as Record<string, unknown>;

  // Images for the reusable gallery (hero + thumbnails + fullscreen + rest).
  const images = property.image_urls ?? [];

  // Header fields — rendered only when present. When granular locality/city
  // exist they form the header place and the combined `location` string is left
  // for the Location section, so no backend field is ever hidden.
  const granularPlace = [property.locality, property.city]
    .filter(Boolean)
    .join(", ");
  const place = granularPlace || property.location;
  const usedLocationString = !granularPlace && Boolean(property.location);

  // Quick Highlights — pick the first present candidate key per spec.
  const highlightConsumed = new Set<string>();
  const highlights = HIGHLIGHT_SPECS.map((spec) => {
    const key = spec.keys.find((k) => !isEmptyValue(record[k]));
    if (!key) return null;
    highlightConsumed.add(key);
    return {
      label: spec.label,
      icon: spec.icon,
      value:
        spec.label === "Area" ? formatArea(record[key] as never) : formatValue(record[key]),
    };
  }).filter(Boolean) as { label: string; icon: LucideIcon; value: string }[];

  // Amenities / features — reuse the backend list fields.
  const amenityKey = ["amenities_mcp", "amenities"].find(
    (k) => !isEmptyValue(record[k])
  );
  const amenities = amenityKey ? asStringList(record[amenityKey]) : [];
  const featureKey = ["features_mcp", "features"].find(
    (k) => !isEmptyValue(record[k])
  );
  const features = featureKey ? asStringList(record[featureKey]) : [];

  // Keys already rendered by a dedicated component — excluded from the dynamic
  // sections and from Additional Information so nothing is shown twice.
  const consumed = new Set<string>([
    "id",
    "image_urls",
    "ap_pjt_url",
    "project_name",
    "builder",
    "locality",
    "city",
    "price",
    "bhk_type",
    "latitude",
    "longitude",
    ...highlightConsumed,
  ]);
  // Only consume the combined `location` string when the header actually shows
  // it; otherwise it falls through to the Location section.
  if (usedLocationString) consumed.add("location");
  if (amenityKey) consumed.add(amenityKey);
  if (featureKey) consumed.add(featureKey);

  // Single-property map input for the reusable interactive map. Only the fields
  // the map renders are picked from the backend record; coordinates decide
  // whether the interactive map or the placeholder fallback is shown.
  const mapProperty: MapProperty = {
    id: property.id,
    latitude: property.latitude,
    longitude: property.longitude,
    price: property.price,
    project_name: property.project_name,
    locality: property.locality,
    city: property.city,
    location: property.location,
    area: property.area,
    bhk_type: property.bhk_type,
    bed: record["bed"] as MapProperty["bed"],
    costpersqft: record["costpersqft"] as MapProperty["costpersqft"],
    image_urls: property.image_urls,
  };
  const hasMapCoords = getMapCoords(mapProperty) !== null;

  const { sections, additional } = categorizeFields(record, consumed);
  const byId = new Map(sections.map((s) => [s.id, s] as const));

  const pricing = byId.get("pricing");
  const overview = byId.get("overview");
  const specifications = byId.get("specifications");
  const project = byId.get("project");
  const nearby = byId.get("nearby");
  const reviews = byId.get("reviews");
  const ratings = byId.get("ratings");
  const location = byId.get("location");
  const insights = byId.get("insights");

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <BackLink />

      {/* 1-2. Hero image + thumbnail gallery + fullscreen viewer + remaining
          images — the single reusable gallery renders every backend image. */}
      <PropertyImageGallery
        images={images}
        alt={property.project_name || `Property ${property.id}`}
        className="sm:aspect-[16/9]"
      />

      {/* 3. Property header. */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <h1 className="font-heading text-2xl font-semibold">
            {property.project_name || "Property"}
          </h1>
          {property.builder && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <Building2 className="size-4 shrink-0" />
              <span className="truncate">{property.builder}</span>
            </p>
          )}
          {place && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <MapPin className="size-4 shrink-0" />
              <span className="truncate">{place}</span>
            </p>
          )}
          <p className="font-mono text-xs text-muted-foreground">{property.id}</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <p className="text-2xl font-semibold">{formatCr(property.price)}</p>
          {property.bhk_type && (
            <Badge variant="secondary">
              <BedDouble /> {property.bhk_type}
            </Badge>
          )}
          <div className="flex items-center gap-2">
            <BookmarkButton id={property.id} />
            <OriginalListingButton url={property.ap_pjt_url} size="sm" />
          </div>
        </div>
      </div>

      {/* 4. Quick Highlights. */}
      {highlights.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {highlights.map((h) => (
            <Card key={h.label} className="gap-0 py-0">
              <CardContent className="flex items-center gap-3 p-4">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <h.icon className="size-4" />
                </span>
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground">{h.label}</p>
                  <p className="truncate font-medium">{h.value}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* 5. Pricing. */}
      {pricing && (
        <Section title="Pricing">
          <FieldSection entries={pricing.entries} />
        </Section>
      )}

      {/* 6. Property Overview. */}
      {overview && (
        <Section title="Property Overview">
          <FieldSection entries={overview.entries} />
        </Section>
      )}

      {/* 7. Property Specifications — grouped cards that adapt to missing fields. */}
      {specifications && (
        <Section title="Property Specifications">
          <SpecificationGroups entries={specifications.entries} />
        </Section>
      )}

      {/* 8. Project Information. */}
      {project && (
        <Section title="Project Information">
          <FieldSection entries={project.entries} />
        </Section>
      )}

      {/* 9. Amenities — responsive icon grid. */}
      {amenities.length > 0 && (
        <Section title="Amenities">
          <AmenityGrid amenities={amenities} />
        </Section>
      )}

      {/* 10. Features. */}
      {features.length > 0 && (
        <Section title="Features">
          <div className="flex flex-wrap gap-1.5">
            {features.map((f) => (
              <Badge key={f} variant="secondary">
                {f}
              </Badge>
            ))}
          </div>
        </Section>
      )}

      {/* 11. Nearby Places. */}
      {nearby && (
        <Section title="Nearby Places">
          <ListFieldSection entries={nearby.entries} />
        </Section>
      )}

      {/* 12. Reviews. */}
      {reviews && (
        <Section title="Reviews">
          <ReviewsSection entries={reviews.entries} />
        </Section>
      )}

      {/* Ratings (part of the reviews group in the backend). */}
      {ratings && (
        <Section title="Ratings">
          <RatingsSection entries={ratings.entries} />
        </Section>
      )}

      {/* 13. Location — the reusable interactive map centred on this property
          (Phase 14.6), with the coordinate placeholder as a graceful fallback
          when the backend has no valid coordinates. */}
      <Section title="Location">
        <div className="space-y-4">
          {hasMapCoords ? (
            <InteractivePropertyMap
              properties={[mapProperty]}
              selectedId={property.id}
              className="h-80 sm:h-96"
            />
          ) : (
            <PropertyLocationMap
              latitude={property.latitude}
              longitude={property.longitude}
              label={place || undefined}
            />
          )}
          {location && <FieldSection entries={location.entries} />}
        </div>
      </Section>

      {/* 14. AI Insights — backend analysis only; never generated here. */}
      {insights && (
        <Section title="AI Insights">
          <InsightsSection entries={insights.entries} />
        </Section>
      )}

      {/* 15. Additional Information — every uncategorized backend field. */}
      {additional.length > 0 && (
        <Section title="Additional Information">
          <FieldSection entries={additional} />
        </Section>
      )}

      {/* 16. View Original Listing — prominent action (reused). */}
      {property.ap_pjt_url && (
        <div className="flex justify-center pt-2">
          <OriginalListingButton url={property.ap_pjt_url} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Header bookmark — reuses the existing saved-property workflow. No new logic.
function BookmarkButton({ id }: { id: string }) {
  const { data: saved } = useSavedProperties();
  const save = useSaveProperty();
  const remove = useRemoveSavedProperty();

  const isSaved = Boolean(saved?.some((p) => p.id === id));
  const pending = save.isPending || remove.isPending;

  return (
    <Button
      variant="outline"
      size="sm"
      aria-pressed={isSaved}
      disabled={pending}
      onClick={() => (isSaved ? remove.mutate(id) : save.mutate(id))}
      title={isSaved ? "Remove from saved" : "Save property"}
    >
      <Bookmark className={cn(isSaved && "fill-primary text-primary")} />
      {isSaved ? "Saved" : "Save"}
    </Button>
  );
}

// ---------------------------------------------------------------------------
// Section wrapper — a titled card, consistent with the design system.
function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 p-6">
        <h2 className="font-heading text-sm font-semibold">{title}</h2>
        <Separator />
        {children}
      </CardContent>
    </Card>
  );
}

// A single label / value pair. List-like values render as chips.
function FieldItem({ entry }: { entry: FieldEntry }) {
  const list = toList(entry.value);
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">{humanizeKey(entry.key)}</p>
      {list ? (
        <div className="flex flex-wrap gap-1">
          {list.map((item) => (
            <Badge key={item} variant="outline">
              {item}
            </Badge>
          ))}
        </div>
      ) : (
        <p className="font-medium break-words">{formatValue(entry.value)}</p>
      )}
    </div>
  );
}

// A generic section body: long text values stack full-width; the rest render in
// a responsive label/value grid.
function FieldSection({ entries }: { entries: FieldEntry[] }) {
  const long: FieldEntry[] = [];
  const short: FieldEntry[] = [];
  for (const entry of entries) {
    if (
      typeof entry.value === "string" &&
      !toList(entry.value) &&
      entry.value.length > 80
    ) {
      long.push(entry);
    } else {
      short.push(entry);
    }
  }

  return (
    <div className="space-y-4">
      {short.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {short.map((entry) => (
            <FieldItem key={entry.key} entry={entry} />
          ))}
        </div>
      )}
      {long.map((entry) => (
        <div key={entry.key} className="space-y-1">
          <p className="text-xs text-muted-foreground">
            {humanizeKey(entry.key)}
          </p>
          <p className="text-sm leading-relaxed text-foreground/90">
            {String(entry.value)}
          </p>
        </div>
      ))}
    </div>
  );
}

// Specifications — chunk fields into groups of three, each an information card.
function SpecificationGroups({ entries }: { entries: FieldEntry[] }) {
  const groups: FieldEntry[][] = [];
  for (let i = 0; i < entries.length; i += 3) {
    groups.push(entries.slice(i, i + 3));
  }
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {groups.map((group, index) => (
        <div key={index} className="space-y-3 rounded-lg border bg-muted/30 p-4">
          {group.map((entry) => (
            <FieldItem key={entry.key} entry={entry} />
          ))}
        </div>
      ))}
    </div>
  );
}

// Nearby places — each field labeled, list values shown as chips.
function ListFieldSection({ entries }: { entries: FieldEntry[] }) {
  return (
    <div className="space-y-4">
      {entries.map((entry) => {
        const list = toList(entry.value);
        return (
          <div key={entry.key} className="space-y-1.5">
            <p className="text-sm font-medium">{humanizeKey(entry.key)}</p>
            {list ? (
              <div className="flex flex-wrap gap-1.5">
                {list.map((item) => (
                  <Badge key={item} variant="outline">
                    <MapPin /> {item}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {formatValue(entry.value)}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

// Reviews — positive / improvement highlights as concise cards.
function ReviewsSection({ entries }: { entries: FieldEntry[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {entries.map((entry) => {
        const lower = entry.key.toLowerCase();
        const positive = /positive|pros|good|like/.test(lower);
        const improve = /improve|cons|bad|things|dislike/.test(lower);
        const Icon = positive
          ? CheckCircle2
          : improve
            ? AlertTriangle
            : Sparkles;
        const tone = positive
          ? "text-emerald-600 dark:text-emerald-400"
          : improve
            ? "text-amber-600 dark:text-amber-400"
            : "text-primary";
        const list = toList(entry.value);
        return (
          <div key={entry.key} className="space-y-2 rounded-lg border p-4">
            <p className="flex items-center gap-2 text-sm font-medium">
              <Icon className={cn("size-4", tone)} />
              {humanizeKey(entry.key)}
            </p>
            {list ? (
              <ul className="space-y-1 text-sm text-muted-foreground">
                {list.map((item) => (
                  <li key={item} className="flex gap-1.5">
                    <span className="text-primary">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">
                {formatValue(entry.value)}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

// Ratings — numeric values render as bars; anything else as a badge.
function RatingsSection({ entries }: { entries: FieldEntry[] }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {entries.map((entry) => {
        const num =
          typeof entry.value === "number"
            ? entry.value
            : Number(entry.value);
        const isNumeric = Number.isFinite(num);
        const scale = num <= 5 ? 5 : num <= 10 ? 10 : 100;
        const pct = isNumeric ? Math.max(0, Math.min(100, (num / scale) * 100)) : 0;
        return (
          <div key={entry.key} className="space-y-1.5">
            <div className="flex items-center justify-between gap-2 text-sm">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <Star className="size-3.5 text-amber-500" />
                {humanizeKey(entry.key)}
              </span>
              {isNumeric ? (
                <span className="font-medium tabular-nums">
                  {num} / {scale}
                </span>
              ) : (
                <Badge variant="outline">{formatValue(entry.value)}</Badge>
              )}
            </div>
            {isNumeric && (
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// AI Insights — render backend analysis text as prose. Never generated here.
function InsightsSection({ entries }: { entries: FieldEntry[] }) {
  return (
    <div className="space-y-4">
      {entries.map((entry) => (
        <div key={entry.key} className="space-y-1">
          <p className="flex items-center gap-1.5 text-sm font-medium">
            <Sparkles className="size-4 text-primary" />
            {humanizeKey(entry.key)}
          </p>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {formatValue(entry.value)}
          </p>
        </div>
      ))}
    </div>
  );
}

// Amenities — responsive icon grid. Icons are matched by keyword with a check
// fallback, so any backend amenity renders.
function AmenityGrid({ amenities }: { amenities: string[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {amenities.map((amenity) => {
        const Icon = amenityIcon(amenity);
        return (
          <div
            key={amenity}
            className="flex items-center gap-2.5 rounded-lg border bg-muted/30 p-3 text-sm"
          >
            <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Icon className="size-4" />
            </span>
            <span className="min-w-0 truncate">{amenity}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers.

// Coerce a backend list field (array or comma-separated string) to a string[].
function asStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((v) => String(v).trim()).filter(Boolean);
  }
  if (typeof value === "string") return splitList(value);
  return [];
}

// Keyword → icon map for the amenity grid. Falls back to a generic icon so any
// backend amenity is rendered.
const AMENITY_ICONS: { match: RegExp; icon: LucideIcon }[] = [
  { match: /gym|fitness/, icon: Dumbbell },
  { match: /pool|swim/, icon: Waves },
  { match: /park|garden|green|lawn/, icon: Trees },
  { match: /parking|car/, icon: Car },
  { match: /power|backup|generator/, icon: Flame },
  { match: /security|cctv|camera|guard|surveillance/, icon: Shield },
  { match: /lift|elevator/, icon: ArrowUpDown },
  { match: /club|community|lounge|hall/, icon: Users },
  { match: /kid|child|play/, icon: Baby },
  { match: /wifi|internet|network/, icon: Wifi },
  { match: /ac|air|ventilation/, icon: Wind },
];

function amenityIcon(name: string): LucideIcon {
  const lower = name.toLowerCase();
  return AMENITY_ICONS.find((a) => a.match.test(lower))?.icon ?? CheckCircle2;
}
