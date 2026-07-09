"use client";

// Reusable Interactive Property Map (Phase 14.6).
//
// The single, reusable map component used across the application (Search Results,
// Property Details, and any future property location view). It consumes only the
// backend latitude / longitude (and optional rich fields) — it never invents
// coordinates, generates fake markers or uses mock data.
//
// This file is the SSR-safe public entry point: Leaflet touches `window`, so the
// actual react-leaflet implementation (property-map-view.tsx) is loaded with
// next/dynamic (ssr: false) behind a loading skeleton. This wrapper also filters
// out properties with missing / invalid coordinates and hides the map gracefully
// (renders nothing) when there is nothing valid to show — so no broken map
// container is ever displayed.

import dynamic from "next/dynamic";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

// Minimal property shape the map needs. A SearchResult / PropertyCardData /
// PropertyDetail all satisfy it structurally (every field is optional except the
// id), so the same component is reused everywhere without adapters.
export type MapProperty = {
  id: string;
  latitude?: number | string | null;
  longitude?: number | string | null;
  price?: number | string | null;
  project_name?: string | null;
  locality?: string | null;
  city?: string | null;
  location?: string | null;
  area?: number | string | null;
  bed?: number | string | null;
  bhk_type?: string | null;
  costpersqft?: number | string | null;
  image_urls?: string[] | null;
};

// A property with parsed, finite coordinates — safe to render as a marker.
export type MapPropertyWithCoords = MapProperty & { lat: number; lng: number };

// Parse a backend coordinate into a finite number, or null when unusable.
// (0,0) is treated as invalid — it is the classic "missing coordinate" sentinel.
function toCoordinate(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(n) ? n : null;
}

// Return the property with parsed coordinates, or null when it cannot be mapped.
export function getMapCoords(
  property: MapProperty
): MapPropertyWithCoords | null {
  const lat = toCoordinate(property.latitude);
  const lng = toCoordinate(property.longitude);
  if (lat === null || lng === null) return null;
  // Guard the valid geographic range and drop the (0,0) null-island sentinel.
  if (Math.abs(lat) > 90 || Math.abs(lng) > 180) return null;
  if (lat === 0 && lng === 0) return null;
  return { ...property, lat, lng };
}

type InteractivePropertyMapProps = {
  properties: MapProperty[];
  // Currently highlighted marker (controlled). Its marker is emphasised.
  selectedId?: string | null;
  // Fired when a marker is selected (id) or deselected (null).
  onSelect?: (id: string | null) => void;
  // Height / layout classes for the map container (e.g. "h-80").
  className?: string;
};

// The react-leaflet implementation is client-only (Leaflet needs `window`), so
// it is code-split and never server-rendered. A skeleton fills the space while
// the map chunk and tiles load.
const PropertyMapView = dynamic(
  () => import("./property-map-view").then((m) => m.PropertyMapView),
  {
    ssr: false,
    loading: () => <Skeleton className="size-full" />,
  }
);

export function InteractivePropertyMap({
  properties,
  selectedId,
  onSelect,
  className,
}: InteractivePropertyMapProps) {
  // Only properties with valid coordinates are mapped. Everything else is
  // ignored — no placeholder markers.
  const points = properties
    .map(getMapCoords)
    .filter((p): p is MapPropertyWithCoords => p !== null);

  // Nothing valid to show → hide the map gracefully (no broken container).
  if (points.length === 0) return null;

  return (
    <div
      className={cn(
        "relative h-[22rem] w-full overflow-hidden rounded-xl border bg-muted shadow-sm",
        className
      )}
    >
      <PropertyMapView
        points={points}
        selectedId={selectedId ?? null}
        onSelect={onSelect}
      />
    </div>
  );
}
