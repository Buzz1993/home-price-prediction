"use client";

// Sticky top bar (Phase 18.1): a floating glass panel matching the sidebar,
// with a rounded global search (⌘K hint), notifications and the signed-in
// user's avatar. The search stays a placeholder — wired up in a later phase —
// and the avatar links to the existing Profile page (no new logic).

import Link from "next/link";
import { Bell, Search } from "lucide-react";

import { useAuth } from "@/components/providers/auth-provider";
import { Button } from "@/components/ui/button";
import { MobileNav } from "./mobile-nav";

export function Navbar() {
  const { user } = useAuth();

  const displayName = user?.name?.trim() || "Guest";
  const initials =
    displayName
      .split(/\s+/)
      .map((part) => part[0])
      .filter(Boolean)
      .slice(0, 2)
      .join("")
      .toUpperCase() || "EM";

  return (
    <header className="glass sticky top-3 z-30 flex h-14 items-center gap-3 rounded-2xl border px-3 shadow-float lg:top-4 lg:px-4">
      <MobileNav />

      {/* Placeholder global search — wired up in a later phase. */}
      <div className="hidden flex-1 sm:block">
        <div className="flex h-9 max-w-md items-center gap-2 rounded-full border bg-background/70 px-4 text-sm text-muted-foreground transition-colors hover:bg-accent">
          <Search className="size-4" />
          <span className="flex-1">Search properties, reports…</span>
          <kbd className="hidden rounded-md border bg-muted px-1.5 py-0.5 font-mono text-[0.65rem] font-medium text-muted-foreground md:inline-block">
            ⌘K
          </kbd>
        </div>
      </div>

      <div className="flex flex-1 items-center justify-end gap-1.5 sm:flex-none">
        <Button variant="ghost" size="icon" aria-label="Notifications">
          <Bell />
        </Button>
        {/* Signed-in user — initials avatar linking to the Profile page. */}
        <Link
          href="/profile"
          aria-label="Open profile"
          title={displayName}
          className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary ring-1 ring-primary/15 transition-shadow hover:ring-primary/40"
        >
          {initials}
        </Link>
      </div>
    </header>
  );
}
