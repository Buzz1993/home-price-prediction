"use client";

// Search Results. Renders ranked properties returned by the backend as a
// responsive grid of reusable PropertyCards and lets the user stage them into
// the evaluation tray — the same interaction as the Streamlit search results
// data-editor, modernised into cards. Each card also exposes a Save toggle
// (Phase 10) so a property can be added to / removed from the saved list; the
// backend owns persistence via the documented saved-properties endpoints.

import { useState } from "react";
import { SearchX } from "lucide-react";

import { PropertyCard } from "@/components/property/property-card";
import { InteractivePropertyMap } from "@/components/property/interactive-property-map";
import { EmptyState } from "@/components/ui/empty-state";
import { cn } from "@/lib/utils";
import {
  useRemoveSavedProperty,
  useSaveProperty,
  useSavedProperties,
} from "@/features/saved/use-saved";
import type { SearchResult } from "@/types/dashboard";
import { useWorkspace } from "./workspace-provider";

export function SearchResultsPanel({ results }: { results: SearchResult[] }) {
  const { tray, toggleTray } = useWorkspace();
  const { data: saved } = useSavedProperties();
  const save = useSaveProperty();
  const remove = useRemoveSavedProperty();

  // Selected marker ↔ card highlight. Selecting a marker highlights the matching
  // card and vice-versa; the map itself is hidden when no result has coordinates.
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const savedIds = new Set((saved ?? []).map((p) => p.id));
  const savePending = save.isPending || remove.isPending;

  const toggleSave = (id: string) => {
    if (savedIds.has(id)) remove.mutate(id);
    else save.mutate(id);
  };

  if (results.length === 0) {
    return (
      <EmptyState
        icon={SearchX}
        title="No matching properties"
        description="No properties matched your search. Try adjusting the location, BHK or amenities."
        className="py-8"
      />
    );
  }

  return (
    <div className="space-y-3">
      {/* Interactive map — renders every result with valid backend coordinates
          and hides gracefully otherwise. Reused across the application. */}
      <InteractivePropertyMap
        properties={results}
        selectedId={selectedId}
        onSelect={setSelectedId}
        className="h-72 sm:h-80"
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {results.map((result) => (
          <div
            key={result.id}
            className={cn(
              "rounded-xl transition-shadow",
              result.id === selectedId && "ring-2 ring-primary ring-offset-2"
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
