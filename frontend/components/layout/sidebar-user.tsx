"use client";

// Shared signed-in user footer pinned to the bottom of a sidebar. Shows the
// authenticated user's name (from the existing AuthProvider — never
// hardcoded) with an initials avatar and links to the Profile page. The user
// model has no role field, so the subtitle uses the documented fallback.
// Phase 18.2: styled as a floating card on the dark purple rail with a Pro
// badge and a settings glyph — purely decorative, still one link to /profile.

import Link from "next/link";
import { Settings } from "lucide-react";

import { useAuth } from "@/components/providers/auth-provider";
import { cn } from "@/lib/utils";

export function SidebarUser({
  className,
  onNavigate,
}: {
  className?: string;
  onNavigate?: () => void;
}) {
  const { user } = useAuth();

  const displayName = user?.name?.trim() || "Guest";
  const displayRole = "EstateMind User";
  const initials =
    displayName
      .split(/\s+/)
      .map((part) => part[0])
      .filter(Boolean)
      .slice(0, 2)
      .join("")
      .toUpperCase() || "EM";

  return (
    <div className={cn("shrink-0 p-3", className)}>
      <Link
        href="/profile"
        onClick={onNavigate}
        className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 transition-colors duration-200 hover:bg-sidebar-accent"
      >
        <span className="bg-brand-gradient flex size-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold text-primary-foreground">
          {initials}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-sm font-medium text-sidebar-foreground">
              {displayName}
            </span>
            <span className="shrink-0 rounded-full bg-brand-secondary/25 px-1.5 py-px text-[0.6rem] font-semibold uppercase tracking-wide text-sidebar-accent-foreground">
              Pro
            </span>
          </span>
          <span className="block truncate text-xs text-sidebar-foreground/60">
            {displayRole}
          </span>
        </span>
        <Settings className="size-4 shrink-0 text-sidebar-foreground/50" />
      </Link>
    </div>
  );
}
