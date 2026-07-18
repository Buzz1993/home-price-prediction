"use client";

// Property Map panel (Phase 15.13). The always-visible third workspace column.
// It renders the conversation's ACCUMULATED property collection on the single
// reusable InteractivePropertyMap (Phase 14.6) and stays in sync with the
// Property Results panel through the shared selectedPropertyId: selecting a
// marker highlights its card, and clicking a card centres the map on it. When
// no property in the conversation has backend coordinates, a friendly empty
// state is shown instead of a broken map container.

import { MapPin } from "lucide-react";

import {
  InteractivePropertyMap,
  getMapCoords,
} from "@/components/property/interactive-property-map";
import { EmptyState } from "@/components/ui/empty-state";
import { useWorkspace } from "./workspace-provider";

export function MapPanel() {
  const { properties, selectedPropertyId, setSelectedPropertyId } =
    useWorkspace();

  const mappable = properties.filter((p) => getMapCoords(p) !== null).length;

  return (
    <div className="flex h-full min-h-0 flex-col" data-tour="property-map">
      {/* Fixed header (Phase 18.3 polish) — small purple icon chip beside the
          title; the map body fills the rest of the column and never scrolls
          away. */}
      <div
        className="flex shrink-0 items-center gap-3 border-b px-4 py-3.5"
        title="View property locations on an interactive map."
      >
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <MapPin className="size-4" />
        </div>
        <div className="min-w-0">
          <h2 className="font-heading text-sm font-semibold">Property Map</h2>
          <p className="truncate text-xs text-muted-foreground">
            {mappable > 0
              ? `${mappable} ${mappable === 1 ? "location" : "locations"}`
              : "Locations from your search"}
          </p>
        </div>
      </div>

      <div className="min-h-0 flex-1 p-3.5">
        {mappable > 0 ? (
          <InteractivePropertyMap
            properties={properties}
            selectedId={selectedPropertyId}
            onSelect={setSelectedPropertyId}
            className="h-full shadow-float"
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <EmptyState
              icon={MapPin}
              title="No map locations yet"
              description="Search for properties and those with a location will appear here on the map."
            />
          </div>
        )}
      </div>
    </div>
  );
}
