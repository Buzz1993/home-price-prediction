"use client";

// Property Results panel (Phase 15.13). Renders the conversation's ACCUMULATED,
// deduplicated property collection (from the workspace provider) as a responsive
// grid of reusable PropertyCards — not just the latest search. Every successful
// search appends its unique properties to this collection, so the grid grows
// (5 -> 10 -> 13) instead of being replaced. The Interactive Property Map renders
// from the same collection, and selecting a card highlights its marker (and vice
// versa) via the shared selectedPropertyId. Each card keeps the tray-staging and
// save toggles; the backend owns search, ranking and persistence.

import { useEffect, useRef } from "react";

import { PropertyCard } from "@/components/property/property-card";
import { cn } from "@/lib/utils";
import {
  useRemoveSavedProperty,
  useSaveProperty,
  useSavedProperties,
} from "@/features/saved/use-saved";
import { useWorkspace } from "./workspace-provider";

export function PropertyResultsPanel() {
  const {
    properties,
    tray,
    toggleTray,
    selectedPropertyId,
    setSelectedPropertyId,
  } = useWorkspace();
  const { data: saved } = useSavedProperties();
  const save = useSaveProperty();
  const remove = useRemoveSavedProperty();

  // Scroll the selected card into view when the selection is driven from the map.
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  useEffect(() => {
    if (!selectedPropertyId) return;
    cardRefs.current[selectedPropertyId]?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [selectedPropertyId]);

  const savedIds = new Set((saved ?? []).map((p) => p.id));
  const savePending = save.isPending || remove.isPending;

  const toggleSave = (id: string) => {
    if (savedIds.has(id)) remove.mutate(id);
    else save.mutate(id);
  };

  if (properties.length === 0) return null;

  return (
    <div className="space-y-3">
      <p className="text-xs font-medium text-muted-foreground">
        {properties.length}{" "}
        {properties.length === 1 ? "property" : "properties"} in this
        conversation
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {properties.map((result) => (
          <div
            key={result.id}
            ref={(el) => {
              cardRefs.current[result.id] = el;
            }}
            onClick={() => setSelectedPropertyId(result.id)}
            className={cn(
              "rounded-xl transition-all",
              result.id === selectedPropertyId &&
                "ring-2 ring-primary ring-offset-2"
            )}
          >
            <PropertyCard
              property={result}
              staged={tray.includes(result.id)}
              onToggleStage={toggleTray}
              saved={savedIds.has(result.id)}
              onToggleSave={toggleSave}
              savePending={savePending}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
