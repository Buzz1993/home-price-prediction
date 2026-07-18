import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { AuthGuard } from "@/components/providers/auth-guard";
import { SearchProvider } from "@/components/providers/search-provider";
import { WorkspaceProvider } from "@/features/dashboard/workspace-provider";
import { TourProvider } from "@/features/onboarding/tour-provider";

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthGuard>
      <WorkspaceProvider>
        <SearchProvider>
          <TourProvider>
            <DashboardLayout>{children}</DashboardLayout>
          </TourProvider>
        </SearchProvider>
      </WorkspaceProvider>
    </AuthGuard>
  );
}
