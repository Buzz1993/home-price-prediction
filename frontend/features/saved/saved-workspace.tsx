"use client";

// Saved Properties page body (Phase 10). Reads the user's saved list from
// GET /saved-properties and renders each entry as a reusable PropertyCard.
// The bookmark toggle removes a property (DELETE /save-property); staging reuses
// the shared evaluation tray so a saved property can flow into comparison,
// analysis or a report. No persistence logic lives here — the backend owns the
// saved list; this only reads it and toggles membership by id, mirroring the
// documented workflow (Property → Save → Saved List).
//
// Phase 18.7: premium presentation polish — hero header with gradient icon,
// premium collection card layout, improved empty state.

import { Bookmark, Heart } from "lucide-react";

import { PropertyCard } from "@/components/property/property-card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { useRemoveSavedProperty, useSavedProperties } from "./use-saved";
import Link from "next/link";

export function SavedWorkspace() {
  const { tray, toggleTray } = useWorkspace();
  const { data, isLoading, isError, isFetching, refetch } =
    useSavedProperties();
  const remove = useRemoveSavedProperty();

  const saved = data ?? [];

  return (
    <div className="space-y-5">
      {/* Premium page hero (Phase 18.7) */}
      <header className="space-y-3">
        <div className="flex items-center gap-4">
          <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 via-accent to-transparent ring-1 ring-primary/10">
            <Heart className="size-7 text-primary" />
          </div>
          <div className="space-y-1">
            <h1 className="font-heading text-3xl font-semibold tracking-tight">
              Saved Properties
            </h1>
            <p className="text-sm text-muted-foreground">
              Your curated collection of favorite properties
            </p>
          </div>
        </div>
      </header>

      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-[420px] rounded-xl" />
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
        <div className="flex min-h-[50vh] items-center justify-center rounded-xl border border-dashed bg-gradient-to-br from-muted/20 to-transparent">
          <EmptyState
            icon={Bookmark}
            title="No saved properties yet"
            description="Save properties from AI Chat search results or property details, then find them here in your collection."
            action={
              <Button asChild variant="default" size="default" className="mt-2">
                <Link href="/dashboard">
                  Browse Properties
                </Link>
              </Button>
            }
          />
        </div>
      )}

      {!isLoading && !isError && saved.length > 0 && (
        <>
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              <span className="font-semibold text-foreground">{saved.length}</span>{" "}
              {saved.length === 1 ? "property" : "properties"} in your collection
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
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
        </>
      )}
    </div>
  );
}
