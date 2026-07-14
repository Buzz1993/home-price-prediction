"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";
import { navItems } from "@/lib/navigation";

// Renders the navigation links (ChatGPT / Cursor-style rows: icon + label,
// rounded, green active state). Shared by the desktop sidebar and the mobile
// drawer. `onNavigate` lets the mobile menu close on selection.
export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <nav className="flex shrink-0 flex-col gap-1.5 px-4">
      {navItems.map((item) => {
        // Dashboard is the single Copilot Workspace entry point, so it stays
        // highlighted on the retained /chat compatibility route too.
        const active =
          item.href === "/dashboard"
            ? pathname === "/dashboard" || pathname === "/chat"
            : pathname === item.href || pathname.startsWith(`${item.href}/`);

        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200",
              active
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
            )}
          >
            <item.icon className="size-4 shrink-0" />
            <span className="truncate">{item.title}</span>
          </Link>
        );
      })}
    </nav>
  );
}
