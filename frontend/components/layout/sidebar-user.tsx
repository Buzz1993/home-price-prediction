"use client";

// Shared signed-in user footer pinned to the bottom of a sidebar. Shows the
// authenticated user's name (from the existing AuthProvider — never
// hardcoded) with an initials avatar and links to the Profile page. The user
// model has no role field, so the subtitle uses the documented fallback.

import Link from "next/link";

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
    <div className={cn("shrink-0 border-t p-3", className)}>
      <Link
        href="/profile"
        onClick={onNavigate}
        className="flex items-center gap-3 rounded-xl px-2 py-2 transition-colors duration-200 hover:bg-muted"
      >
        <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
          {initials}
        </span>
        <span className="min-w-0">
          <span className="block truncate text-sm font-medium">
            {displayName}
          </span>
          <span className="block truncate text-xs text-muted-foreground">
            {displayRole}
          </span>
        </span>
      </Link>
    </div>
  );
}
