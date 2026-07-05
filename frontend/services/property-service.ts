// Property details API call. Thin wrapper over the documented GET /property/{id}
// endpoint — the backend owns the drill-down query and field whitelist. See
// project_docs/03_API.md and src/mcp/tools/property_tools.py -> get_property_details.

import { apiRequest } from "@/lib/api-client";
import type { NotFoundResponse, PropertyDetail } from "@/types/dashboard";

// The backend returns the property record, or `{ error }` (HTTP 200) when the id
// has no match. Callers narrow on the `error` field.
export function getPropertyDetails(
  id: string
): Promise<PropertyDetail | NotFoundResponse> {
  return apiRequest<PropertyDetail | NotFoundResponse>(
    `/property/${encodeURIComponent(id)}`
  );
}
