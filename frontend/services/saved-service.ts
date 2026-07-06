// Saved Properties API calls. Thin wrappers over the documented Saved Properties
// endpoints (GET /saved-properties, POST /save-property, DELETE /save-property —
// see project_docs/03_API.md). The backend owns persistence of the saved list;
// the frontend only reads it and toggles membership by property id.
//
// Backend limitation: the EstateMind Copilot API (src/api) currently exposes only
// the /analysis/* endpoints — the saved-properties endpoints are not wired yet.
// These wrappers target the documented contract so live data flows once the
// endpoints are exposed, matching the Phase 4–9 pattern. No API is invented and
// no business logic lives in the frontend.

import { apiRequest } from "@/lib/api-client";
import type { SavedProperty } from "@/types/dashboard";

// GET /saved-properties — retrieve the user's saved property list.
export function getSavedProperties(): Promise<SavedProperty[]> {
  return apiRequest<SavedProperty[]>("/saved-properties");
}

// POST /save-property — add a property to the saved list.
export function saveProperty(id: string): Promise<unknown> {
  return apiRequest("/save-property", {
    method: "POST",
    body: { property_id: id },
  });
}

// DELETE /save-property — remove a property from the saved list.
export function removeSavedProperty(id: string): Promise<unknown> {
  return apiRequest("/save-property", {
    method: "DELETE",
    body: { property_id: id },
  });
}
