"use client";

// Search Results. Renders ranked properties returned by the backend as a
// responsive grid of reusable PropertyCards and lets the user stage them into
// the evaluation tray — the same interaction as the Streamlit search results
// data-editor, modernised into cards. Each card also exposes a Save toggle
// (Phase 10) so a property can be added to / removed from the saved list; the
// backend owns persistence via the documented saved-properties endpoints.

import { PropertyCard } from "@/components/property/property-card";
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

  const savedIds = new Set((saved ?? []).map((p) => p.id));
  const savePending = save.isPending || remove.isPending;

  const toggleSave = (id: string) => {
    if (savedIds.has(id)) remove.mutate(id);
    else save.mutate(id);
  };

  if (results.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No properties matched your search. Try adjusting the location, BHK or
        amenities.
      </p>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {results.map((result) => (
        <PropertyCard
          key={result.id}
          property={result}
          staged={tray.includes(result.id)}
          onToggleStage={toggleTray}
          saved={savedIds.has(result.id)}
          onToggleSave={toggleSave}
          savePending={savePending}
        />
      ))}
    </div>
  );
}
