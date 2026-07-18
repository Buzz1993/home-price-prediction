"use client";

// Sticky top bar (Phase 18.1): a floating glass panel matching the sidebar,
// with notifications and the signed-in user's avatar.
//
// Phase 18.9: the search is a REAL input wired to the shared SearchProvider,
// and it only appears on routes with searchable content (currently the Reports
// page, where it filters the report history). Pages without meaningful
// searchable content (Profile, Saved, Comparison, AI Analysis, …) show no
// search box at all — no fake search anywhere. The Copilot workspace routes
// render their own full-bleed shell without this navbar.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, Search } from "lucide-react";

import { useAuth } from "@/components/providers/auth-provider";
import { useGlobalSearch } from "@/components/providers/search-provider";
import { Button } from "@/components/ui/button";
import { MobileNav } from "./mobile-nav";

// Routes where the global search has real content to filter, with a matching
// placeholder. Search is hidden everywhere else.
const SEARCHABLE_ROUTES: Record<string, string> = {
  "/reports": "Search reports…",
};

export function Navbar() {
  const { user } = useAuth();
  const pathname = usePathname();
  const { query, setQuery } = useGlobalSearch();

  const placeholder = SEARCHABLE_ROUTES[pathname];

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

      {/* Functional search — only on routes with searchable content. Full
          width on mobile (Phase 20), capped on larger screens. */}
      {placeholder && (
        <div className="min-w-0 flex-1">
          <label className="flex h-9 items-center gap-2 rounded-full border bg-background/70 px-4 text-sm transition-colors focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/20 hover:bg-accent/40 sm:max-w-md">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={placeholder}
              aria-label={placeholder}
              className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-muted-foreground"
            />
          </label>
        </div>
      )}

      <div className="flex flex-1 items-center justify-end gap-1.5">
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
