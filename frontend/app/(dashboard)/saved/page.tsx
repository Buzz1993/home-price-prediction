import { SavedWorkspace } from "@/features/saved/saved-workspace";

// Saved Properties (Phase 10): a dedicated page that lists the user's saved
// properties (GET /saved-properties) and lets them remove one
// (DELETE /save-property) or stage it in the shared evaluation tray. Reuses the
// PropertyCard and tray — no persistence logic is duplicated in the frontend.
export default function SavedPage() {
  return <SavedWorkspace />;
}
