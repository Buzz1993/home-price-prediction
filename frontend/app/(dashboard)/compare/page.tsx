import { ComparisonWorkspace } from "@/features/comparison/comparison-workspace";

// Property Comparison (Phase 7): a dedicated page that compares the properties
// staged in the shared evaluation tray via POST /compare and renders the score
// cards, AI recommendation and ranking table. Tray + comparison rendering are
// reused from the Copilot workspace — no comparison logic is duplicated.
export default function ComparePage() {
  return <ComparisonWorkspace />;
}
