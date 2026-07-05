"use client";

// TanStack Query hook for a single property's details. Keeps fetching state
// (loading / success / error) out of the view. Business logic stays in the
// backend; this only caches the response by id.

import { useQuery } from "@tanstack/react-query";

import { getPropertyDetails } from "@/services/property-service";

export function usePropertyDetails(id: string) {
  return useQuery({
    queryKey: ["property", id],
    queryFn: () => getPropertyDetails(id),
    enabled: Boolean(id),
  });
}
