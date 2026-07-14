"use client";

// Property Results panel (Phase 15.13). Renders the property cards returned by ONE
// search — the `results` owned by that specific assistant message
// (message.response.content) — as a responsive grid of reusable PropertyCards.
// Each assistant search response permanently owns and displays its own cards;
// they are never moved to, or replaced by, another message.
//
// This is the MESSAGE-LEVEL view. The CONVERSATION-LEVEL accumulated,
// deduplicated collection (conversation.properties) is used ONLY by the
// Interactive Property Map (see MapPanel), which shows every unique property
// discovered across the whole conversation. Selecting a card highlights its
// marker (and vice versa) through the shared selectedPropertyId; each card keeps
// the tray-staging and save toggles. The backend owns search, ranking and
// persistence.

import { useEffect, useRef } from "react";

import { PropertyCard } from "@/components/property/property-card";
import { cn } from "@/lib/utils";
import {
  useRemoveSavedProperty,
  useSaveProperty,
  useSavedProperties,
} from "@/features/saved/use-saved";
import type { SearchResult } from "@/types/dashboard";
import { useWorkspace } from "./workspace-provider";

export function PropertyResultsPanel({ results }: { results: SearchResult[] }) {
  const { tray, toggleTray, selectedPropertyId, setSelectedPropertyId } =
    useWorkspace();
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

  if (results.length === 0) return null;

  return (
    <div className="space-y-3">
      <p className="text-xs font-medium text-muted-foreground">
        {results.length} {results.length === 1 ? "property" : "properties"} for
        this search
      </p>
      {/* Single column so the horizontal cards get the full width — this keeps
          each card short and lets more properties show without scrolling. */}
      <div className="grid gap-3">
        {results.map((result) => (
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
              horizontal
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
