import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { SearchProvider } from "@/components/providers/search-provider";
import { WorkspaceProvider } from "@/features/dashboard/workspace-provider";

// Applies the app shell (sidebar, navbar, responsive navigation) to every
// protected page in this route group. WorkspaceProvider lives here (not inside
// CopilotWorkspace) so the evaluation tray and chat state persist across the
// Dashboard, AI Chat and Property Comparison routes — matching the Streamlit
// session_state tray. SearchProvider (Phase 18.9) carries the navbar's search
// query to the page that owns the searchable content.
export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <WorkspaceProvider>
      <SearchProvider>
        <DashboardLayout>{children}</DashboardLayout>
      </SearchProvider>
    </WorkspaceProvider>
  );
}
