// Reusable property location container (Phase 14.4).
//
// This is a lightweight placeholder that consumes the backend latitude /
// longitude only — the full interactive property map is Phase 14.6 and is NOT
// implemented here. It renders a map-style panel with the coordinates and
// external "open in maps" links (Google Maps / OpenStreetMap) when valid
// coordinates exist, and a graceful fallback otherwise. Presentational only.

import { ExternalLink, MapPin } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type PropertyLocationMapProps = {
  latitude?: number | string | null;
  longitude?: number | string | null;
  // Human-readable place used as the marker label (e.g. locality, city).
  label?: string;
  className?: string;
};

// Parse a backend coordinate into a finite number, or null when unusable.
function toCoordinate(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(n) ? n : null;
}

export function PropertyLocationMap({
  latitude,
  longitude,
  label,
  className,
}: PropertyLocationMapProps) {
  const lat = toCoordinate(latitude);
  const lng = toCoordinate(longitude);
  const hasCoords = lat !== null && lng !== null;

  const googleUrl = hasCoords
    ? `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`
    : null;
  const osmUrl = hasCoords
    ? `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lng}#map=16/${lat}/${lng}`
    : null;

  return (
    <div
      className={cn(
        "relative flex min-h-44 flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border bg-muted/40 p-6 text-center",
        // Subtle map-like grid backdrop so the placeholder reads as a map.
        "bg-[linear-gradient(to_right,theme(colors.border)_1px,transparent_1px),linear-gradient(to_bottom,theme(colors.border)_1px,transparent_1px)] bg-[size:24px_24px]",
        className
      )}
    >
      <span className="flex size-11 items-center justify-center rounded-full bg-primary/10 text-primary">
        <MapPin className="size-5" />
      </span>

      {hasCoords ? (
        <>
          <div className="space-y-0.5">
            {label && <p className="font-medium">{label}</p>}
            <p className="font-mono text-xs text-muted-foreground tabular-nums">
              {lat}, {lng}
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2">
            <Button asChild variant="outline" size="sm">
              <a href={googleUrl!} target="_blank" rel="noopener noreferrer">
                Google Maps <ExternalLink />
              </a>
            </Button>
            <Button asChild variant="outline" size="sm">
              <a href={osmUrl!} target="_blank" rel="noopener noreferrer">
                OpenStreetMap <ExternalLink />
              </a>
            </Button>
          </div>
        </>
      ) : (
        <p className="max-w-xs text-sm text-muted-foreground">
          {label || "Location coordinates are not available for this property."}
        </p>
      )}
    </div>
  );
}
