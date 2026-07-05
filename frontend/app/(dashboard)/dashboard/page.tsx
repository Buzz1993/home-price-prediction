import { CopilotWorkspace } from "@/features/dashboard/copilot-workspace";

// Dashboard: the EstateMind Copilot workspace (chat + evaluation tray),
// reproducing the Streamlit copilot layout. The shell is shared with the AI
// Chat page via CopilotWorkspace.
export default function DashboardPage() {
  return <CopilotWorkspace />;
}
