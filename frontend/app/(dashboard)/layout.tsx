import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { WorkspaceProvider } from "@/features/dashboard/workspace-provider";

// Applies the app shell (sidebar, navbar, responsive navigation) to every
// protected page in this route group. WorkspaceProvider lives here (not inside
// CopilotWorkspace) so the evaluation tray and chat state persist across the
// Dashboard, AI Chat and Property Comparison routes — matching the Streamlit
// session_state tray.
export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <WorkspaceProvider>
      <DashboardLayout>{children}</DashboardLayout>
    </WorkspaceProvider>
  );
}
