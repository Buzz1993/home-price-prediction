import { Navbar } from "./navbar";
import { Sidebar } from "./sidebar";

// Application shell for protected pages: a fixed sidebar on desktop, a top
// navbar (with mobile navigation), and a scrollable content area.
export function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-dvh">
      <Sidebar className="sticky top-0 hidden h-dvh lg:flex" />
      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar />
        <main className="flex-1 p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
