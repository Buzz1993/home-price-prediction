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

const SAVED_KEY = ["saved-properties"] as const;

export function useSavedProperties() {
  return useQuery({
    queryKey: SAVED_KEY,
    queryFn: getSavedProperties,
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
