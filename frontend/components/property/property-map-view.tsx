"use client";

// react-leaflet implementation behind the InteractivePropertyMap wrapper (Phase
// 14.6). Client-only — loaded via next/dynamic (ssr: false) so Leaflet only runs
// in the browser. Presentational only: it renders the backend points as price
// markers on an OpenStreetMap tile layer, clusters them for large datasets, auto
// fits the viewport to every marker, and shows a rich preview card (reusing the
// existing format helpers, icons and navigation) when a marker is selected.

import "leaflet/dist/leaflet.css";
import "leaflet.markercluster/dist/MarkerCluster.css";
import "leaflet.markercluster";

import { useEffect, useMemo } from "react";
import Link from "next/link";
import L from "leaflet";
import { createPathComponent } from "@react-leaflet/core";
import { MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";
import { ArrowUpRight, BedDouble, ImageOff, MapPin, Ruler } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  formatArea,
  formatCr,
  formatPerSqft,
  formatPriceLabel,
} from "@/features/dashboard/format";
import type { MapPropertyWithCoords } from "./interactive-property-map";

type PropertyMapViewProps = {
  points: MapPropertyWithCoords[];
  selectedId: string | null;
  onSelect?: (id: string | null) => void;
};

// India centroid — a neutral initial view; FitBounds immediately reframes it to
// the actual markers, so this is never really seen.
const INITIAL_CENTER: [number, number] = [22.9734, 78.6569];
const INITIAL_ZOOM = 5;

export function PropertyMapView({
  points,
  selectedId,
  onSelect,
}: PropertyMapViewProps) {
  return (
    <MapContainer
      center={INITIAL_CENTER}
      zoom={INITIAL_ZOOM}
      scrollWheelZoom
      className="z-0 size-full"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Marker clustering keeps rendering efficient for large datasets. */}
      <ClusterGroup>
        {points.map((p) => (
          <Marker
            key={p.id}
            position={[p.lat, p.lng]}
            icon={priceIcon(p, p.id === selectedId)}
            zIndexOffset={p.id === selectedId ? 1000 : 0}
            eventHandlers={{
              // Drive selection from the popup lifecycle (not click) so switching
              // markers is deterministic: Leaflet closes the previous popup then
              // opens the new one, giving null → id in order. Using click would
              // race with the previous marker's popupclose.
              popupopen: () => onSelect?.(p.id),
              popupclose: () => onSelect?.(null),
            }}
          >
            <Popup className="em-map-popup" minWidth={220} maxWidth={260}>
              <PropertyPreview property={p} />
            </Popup>
          </Marker>
        ))}
      </ClusterGroup>

      <FitBounds points={points} />
      <ResponsiveResize />
    </MapContainer>
  );
}

// ---------------------------------------------------------------------------
// Marker icon — a compact pill showing only the formatted price. The selected
// marker is emphasised with the primary colour. Built as an HTML divIcon so the
// design-system utility classes apply; translate(-50%,-50%) centres the variable
// width pill on the exact coordinate.
function priceIcon(property: MapPropertyWithCoords, selected: boolean): L.DivIcon {
  const label = formatPriceLabel(property.price);
  const tone = selected
    ? "bg-primary text-primary-foreground border-primary"
    : "bg-background text-foreground border-primary/60 hover:border-primary";
  return L.divIcon({
    className: "em-price-marker",
    iconSize: [0, 0],
    html: `<div style="transform:translate(-50%,-50%)" class="inline-flex -translate-x-1/2 -translate-y-1/2 cursor-pointer items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-semibold tabular-nums shadow-md ${tone}">${label}</div>`,
  });
}

// Cluster bubble — a green circle with the child count, matching the design
// system. Replaces the plugin's default styling entirely.
function clusterIcon(count: number): L.DivIcon {
  const size = count < 10 ? 34 : count < 100 ? 40 : 48;
  return L.divIcon({
    className: "em-cluster-marker",
    iconSize: [size, size],
    html: `<div style="width:${size}px;height:${size}px" class="flex items-center justify-center rounded-full border-2 border-white bg-primary text-xs font-semibold text-primary-foreground shadow-md ring-2 ring-primary/30">${count}</div>`,
  });
}

// ---------------------------------------------------------------------------
// A thin react-leaflet binding around leaflet.markercluster. Children (<Marker>)
// register into the cluster group instead of the map via the layerContainer
// context — the same technique react-leaflet uses for LayerGroup. Works with
// React 19 (no React-18-pinned third-party cluster wrapper needed).
const ClusterGroup = createPathComponent<L.MarkerClusterGroup, object>(
  function createClusterGroup(_props, ctx) {
    const group = L.markerClusterGroup({
      chunkedLoading: true,
      showCoverageOnHover: false,
      maxClusterRadius: 60,
      iconCreateFunction: (cluster) => clusterIcon(cluster.getChildCount()),
    });
    return {
      instance: group,
      context: { ...ctx, layerContainer: group },
    };
  }
);

// ---------------------------------------------------------------------------
// Auto fit the viewport so every marker is visible, re-fitting whenever the set
// of points changes. A single point centres at a comfortable zoom instead.
function FitBounds({ points }: { points: MapPropertyWithCoords[] }) {
  const map = useMap();

  // Stable key so the effect only re-fits when the coordinates actually change,
  // not on every render.
  const boundsKey = useMemo(
    () => points.map((p) => `${p.lat},${p.lng}`).join("|"),
    [points]
  );

  useEffect(() => {
    if (points.length === 0) return;
    if (points.length === 1) {
      map.setView([points[0].lat, points[0].lng], 14, { animate: false });
      return;
    }
    const bounds = L.latLngBounds(points.map((p) => [p.lat, p.lng]));
    map.fitBounds(bounds, { padding: [48, 48], maxZoom: 16 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, boundsKey]);

  return null;
}

// ---------------------------------------------------------------------------
// Keep Leaflet's internal size in sync with the responsive container. Without
// invalidateSize the tiles can render grey when the container is resized (e.g.
// grid reflow, breakpoint change, mobile rotation).
function ResponsiveResize() {
  const map = useMap();
  useEffect(() => {
    const container = map.getContainer();
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(container);
    // Ensure a correct first paint once the container has its final size.
    map.invalidateSize();
    return () => observer.disconnect();
  }, [map]);
  return null;
}

// ---------------------------------------------------------------------------
// Rich preview shown in the selected marker's popup. Only renders backend fields
// that exist and reuses the existing format helpers, icons and navigation. The
// full image gallery lives on the details page reached via Read More.
function PropertyPreview({ property }: { property: MapPropertyWithCoords }) {
  const image = property.image_urls?.find(Boolean);
  const place =
    [property.locality, property.city].filter(Boolean).join(", ") ||
    property.location ||
    "";
  const bhk = property.bhk_type || (property.bed ? `${property.bed} BHK` : "");

  return (
    <div className="w-full space-y-2">
      {/* Primary image (image_urls[0]) with a graceful fallback. */}
      <div className="relative aspect-[16/9] w-full overflow-hidden rounded-md bg-muted">
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={image}
            alt={property.project_name || `Property ${property.id}`}
            loading="lazy"
            decoding="async"
            className="size-full object-cover"
          />
        ) : (
          <div className="flex size-full items-center justify-center text-muted-foreground">
            <ImageOff className="size-6" />
          </div>
        )}
      </div>

      {/* Price + BHK. */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-base font-semibold leading-none">
          {formatCr(property.price)}
        </p>
        {bhk && (
          <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            <BedDouble className="size-3" /> {bhk}
          </span>
        )}
      </div>

      {/* Project name. */}
      {property.project_name && (
        <p className="truncate text-sm font-medium leading-tight">
          {property.project_name}
        </p>
      )}

      {/* Locality / city. */}
      {place && (
        <p className="flex items-center gap-1 text-xs text-muted-foreground">
          <MapPin className="size-3 shrink-0" />
          <span className="truncate">{place}</span>
        </p>
      )}

      {/* Area + cost per sqft. */}
      {(property.area || property.costpersqft) && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          {property.area != null && property.area !== "" && (
            <span className="flex items-center gap-1">
              <Ruler className="size-3 shrink-0" />
              <span className="tabular-nums">{formatArea(property.area)}</span>
            </span>
          )}
          {property.costpersqft != null && property.costpersqft !== "" && (
            <span className="tabular-nums">
              {formatPerSqft(property.costpersqft)}
            </span>
          )}
        </div>
      )}

      {/* Read More — reuses the same navigation as the Property Cards. */}
      <Button asChild size="sm" className="w-full">
        <Link href={`/property/${property.id}`}>
          Read More <ArrowUpRight />
        </Link>
      </Button>
    </div>
  );
}
