"use client";

// Global app sidebar for the analysis-style pages (AI Analysis, Saved,
// Reports, Profile, Property details). Deliberately minimal so those
// workspaces stay distraction-free: brand + badge, primary navigation and the
// signed-in user — no chat controls. The Dashboard's chat workspace has its
// own richer ConversationSidebar (New Chat / Search / Recent) in
// features/dashboard. Shared pieces (header, nav, user footer) are reused by
// both rails.

import { cn } from "@/lib/utils";
import { SidebarHeader } from "./sidebar-header";
import { SidebarNav } from "./sidebar-nav";
import { SidebarUser } from "./sidebar-user";

export function Sidebar({
  className,
  onNavigate,
}: {
  className?: string;
  // Lets the mobile drawer close after any selection.
  onNavigate?: () => void;
}) {
  return (
    <aside
      className={cn(
        "flex w-[280px] flex-col border-r bg-sidebar text-sidebar-foreground",
        className
      )}
    >
      <SidebarHeader />
      <SidebarNav onNavigate={onNavigate} />
      {/* Whitespace pushes the user footer to the bottom. */}
      <div className="flex-1" />
      <SidebarUser onNavigate={onNavigate} />
    </aside>
  );
}
