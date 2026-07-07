"use client";

// Saved Properties data hooks. One query for the saved list and two mutations to
// toggle membership (save / remove). Both mutations invalidate the list so the UI
// stays in sync after a change. All persistence lives in the backend; these only
// track request state and cache the response.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getSavedProperties,
  removeSavedProperty,
  saveProperty,
} from "@/services/saved-service";
import { getPropertyDetails } from "@/services/property-service";
import type { PropertyCardData, PropertyDetail } from "@/types/dashboard";

const SAVED_KEY = ["saved-properties"] as const;

// The backend stores only saved property IDs, so each id is resolved to its full
// record (GET /property/{id}) and mapped onto the card shape. Ids that no longer
// resolve are dropped so one missing property never breaks the whole list.
async function fetchSavedProperties(): Promise<PropertyCardData[]> {
  const ids = await getSavedProperties();
  const records = await Promise.all(
    ids.map((id) =>
      getPropertyDetails(id)
        .then((data) => ("error" in data ? null : data))
        .catch(() => null)
    )
  );
  return records
    .filter((data): data is PropertyDetail => data !== null)
    .map((data) => ({
      id: data.id,
      price: data.price,
      bhk_type: data.bhk_type,
      location: data.location,
      amenities_mcp: data.amenities_mcp,
      why_recommended: data.analysis_msg || undefined,
    }));
}

export function useSavedProperties() {
  return useQuery({
    queryKey: SAVED_KEY,
    queryFn: fetchSavedProperties,
  });
}

export function useSaveProperty() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: saveProperty,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: SAVED_KEY }),
  });
}

export function useRemoveSavedProperty() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: removeSavedProperty,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: SAVED_KEY }),
  });
}
