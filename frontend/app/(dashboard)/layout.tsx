import { DashboardLayout } from "@/components/layout/dashboard-layout";

// Applies the app shell (sidebar, navbar, responsive navigation) to every
// protected page in this route group.
export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <DashboardLayout>{children}</DashboardLayout>;
}
