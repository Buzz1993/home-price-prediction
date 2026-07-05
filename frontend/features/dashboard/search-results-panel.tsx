"use client";

// Search Results. Renders ranked properties returned by the backend as a
// responsive grid of reusable PropertyCards and lets the user stage them into
// the evaluation tray — the same interaction as the Streamlit search results
// data-editor, modernised into cards.

import { PropertyCard } from "@/components/property/property-card";
import type { SearchResult } from "@/types/dashboard";
import { useWorkspace } from "./workspace-provider";

export function SearchResultsPanel({ results }: { results: SearchResult[] }) {
  const { tray, toggleTray } = useWorkspace();

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
        />
      ))}
    </div>
  );
}
