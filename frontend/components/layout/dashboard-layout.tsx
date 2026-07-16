"use client";

// Application shell for protected pages. Most pages get the standard chrome: a
// fixed global sidebar, a top navbar (with mobile navigation) and a padded,
// scrollable content area.
//
// The Copilot workspace routes (/dashboard, /chat) instead run full-bleed so
// their own four-column WorkspaceShell (Conversation Sidebar | Chat | Map |
// Tray) fills the viewport — the global navigation is folded into the
// conversation sidebar's footer there, so no second sidebar is shown.

import { usePathname } from "next/navigation";

import { Navbar } from "./navbar";
import { Sidebar } from "./sidebar";

// Routes that render their own full-screen workspace (no global chrome).
const WORKSPACE_ROUTES = new Set(["/dashboard", "/chat"]);

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (WORKSPACE_ROUTES.has(pathname)) {
    return <div className="h-dvh overflow-hidden">{children}</div>;
  }

  return (
    // Floating chrome (Phase 18.1): the page sits on the soft slate canvas
    // with a gutter around the sidebar so it reads as a detached glass panel.
    <div className="flex min-h-dvh gap-3 p-3 lg:gap-4 lg:p-4">
      <Sidebar className="sticky top-3 hidden h-[calc(100dvh-1.5rem)] lg:top-4 lg:flex lg:h-[calc(100dvh-2rem)]" />
      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar />
        <main className="flex-1 py-4 lg:py-6">{children}</main>
      </div>
    </div>
  );
}
