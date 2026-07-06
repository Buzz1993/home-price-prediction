import { AnalysisWorkspace } from "@/features/analysis/analysis-workspace";

// AI Analysis (Phase 8): a dedicated page that runs the per-property analyses
// (price prediction, rental, valuation, risk, investment advice, negotiation)
// on the properties staged in the shared evaluation tray. Tray + result
// renderers are reused from the Copilot workspace — no analysis logic is
// duplicated in the frontend.
export default function AnalysisPage() {
  return <AnalysisWorkspace />;
}
