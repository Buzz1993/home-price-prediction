import { CompareWorkspace } from "@/features/comparison/compare-workspace";

// Property Comparison (Phase 17.0): a dedicated page for professional
// side-by-side comparison of 2–3 properties staged in the shared evaluation
// tray. Reuses the existing backend comparison and analysis endpoints — all
// scoring, ranking and analysis stays in the backend; this page only presents
// the results row-wise with smart highlighting.
export default function ComparePage() {
  return <CompareWorkspace />;
}
