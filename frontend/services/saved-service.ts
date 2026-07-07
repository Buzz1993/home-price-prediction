// Saved Properties API calls. Thin wrappers over the Saved Properties endpoints
// (GET /saved-properties, POST /save-property, DELETE /save-property — see
// src/api/saved_api.py). The backend owns persistence of the saved list; the
// frontend only reads it and toggles membership by property id.

import { apiRequest } from "@/lib/api-client";

// GET /saved-properties — retrieve the user's saved property IDs. The backend
// stores only IDs and returns them under `saved_properties`.
export function getSavedProperties(): Promise<string[]> {
  return apiRequest<{ saved_properties: string[] }>("/saved-properties").then(
    (res) => res.saved_properties ?? []
  );
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
