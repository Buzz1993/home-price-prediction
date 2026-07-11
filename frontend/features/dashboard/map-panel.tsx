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
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="font-heading text-sm font-semibold">Property Map</h2>
          <p className="text-xs text-muted-foreground">
            {mappable > 0
              ? `${mappable} ${mappable === 1 ? "location" : "locations"}`
              : "Locations from your search"}
          </p>
        </div>
      </div>

      <div className="min-h-0 flex-1 p-3">
        {mappable > 0 ? (
          <InteractivePropertyMap
            properties={properties}
            selectedId={selectedPropertyId}
            onSelect={setSelectedPropertyId}
            className="h-full"
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
