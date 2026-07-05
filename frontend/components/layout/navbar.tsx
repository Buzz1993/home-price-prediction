import { Bell, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { MobileNav } from "./mobile-nav";

export function Navbar() {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b bg-background px-4 lg:px-6">
      <MobileNav />

      {/* Placeholder search — wired up in a later phase */}
      <div className="hidden flex-1 sm:block">
        <div className="flex h-9 max-w-sm items-center gap-2 rounded-lg border bg-muted/50 px-3 text-sm text-muted-foreground">
          <Search className="size-4" />
          Search…
        </div>
      </div>

      <div className="flex flex-1 items-center justify-end gap-2 sm:flex-none">
        <Button variant="ghost" size="icon" aria-label="Notifications">
          <Bell />
        </Button>
        {/* Placeholder user avatar — profile menu added in a later phase */}
        <div className="size-8 rounded-full bg-muted" aria-hidden />
      </div>
    </header>
  );
}
