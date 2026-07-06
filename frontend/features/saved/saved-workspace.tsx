"use client";

// Saved Properties page body (Phase 10). Reads the user's saved list from
// GET /saved-properties and renders each entry as a reusable PropertyCard.
// The bookmark toggle removes a property (DELETE /save-property); staging reuses
// the shared evaluation tray so a saved property can flow into comparison,
// analysis or a report. No persistence logic lives here — the backend owns the
// saved list; this only reads it and toggles membership by id, mirroring the
// documented workflow (Property → Save → Saved List).

import { Bookmark } from "lucide-react";

import { PropertyCard } from "@/components/property/property-card";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { useRemoveSavedProperty, useSavedProperties } from "./use-saved";

export function SavedWorkspace() {
  const { tray, toggleTray } = useWorkspace();
  const { data, isLoading, isError, isFetching, refetch } =
    useSavedProperties();
  const remove = useRemoveSavedProperty();

  const saved = data ?? [];

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="font-heading text-lg font-semibold">Saved Properties</h1>
        <p className="text-sm text-muted-foreground">
          Properties you have saved. Remove one with the bookmark, or stage it in
          the tray to compare, analyze or report on it.
        </p>
      </header>

      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-48 rounded-xl" />
          ))}
        </div>
      )}

      {isError && (
        <ErrorState
          title="Could not load your saved properties"
          description="Something went wrong while loading your saved list. Please try again."
          onRetry={() => refetch()}
          retrying={isFetching}
        />
      )}

      {!isLoading && !isError && saved.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed py-16 text-center">
          <Bookmark className="size-8 text-muted-foreground" />
          <p className="max-w-sm text-sm text-muted-foreground">
            You have no saved properties yet. Save properties from the AI Chat
            search results, then find them here.
          </p>
        </div>
      )}

      {!isLoading && !isError && saved.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {saved.map((property) => (
            <PropertyCard
              key={property.id}
              property={property}
              staged={tray.includes(property.id)}
              onToggleStage={toggleTray}
              saved
              onToggleSave={(id) => remove.mutate(id)}
              savePending={remove.isPending}
            />
          ))}
        </div>
      )}
    </div>
  );
}
