import { ReportsWorkspace } from "@/features/reports/reports-workspace";

// Reports (Phase 9): a dedicated page that generates an AI report for the
// properties staged in the shared evaluation tray via POST /report, previews and
// downloads it, then shares it to a phone number via POST /report/share. Tray is
// reused from the Copilot workspace — no report logic is duplicated.
export default function ReportsPage() {
  return <ReportsWorkspace />;
}
