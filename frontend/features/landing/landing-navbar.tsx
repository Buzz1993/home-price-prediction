import Link from "next/link";
import { Building2 } from "lucide-react";

import { Button } from "@/components/ui/button";

// Public top navigation for the landing page. Distinct from the dashboard
// navbar (which is for authenticated app pages).
export function LandingNavbar() {
  return (
    <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur supports-backdrop-filter:bg-background/60">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 lg:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 font-heading text-lg font-semibold"
        >
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Building2 className="size-5" />
          </span>
          EstateMind
        </Link>

        <nav className="flex items-center gap-2">
          <Button asChild variant="ghost" size="lg">
            <Link href="/login">Log in</Link>
          </Button>
          <Button asChild size="lg">
            <Link href="/signup">Get started</Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}
