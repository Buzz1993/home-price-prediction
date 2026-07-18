"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";
import { navItems } from "@/lib/navigation";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Tooltip descriptions for each nav route.
const NAV_TOOLTIPS: Record<string, string> = {
  "/dashboard": "Search and explore AI-recommended properties.",
  "/analysis": "Run investment, rental and valuation analyses.",
  "/compare": "Compare multiple shortlisted properties.",
  "/saved": "View your bookmarked properties.",
  "/reports": "Generate and manage AI investment reports.",
  "/profile": "Manage your account settings.",
};

// data-tour IDs for each nav route (used by the onboarding tour).
const NAV_TOUR_IDS: Record<string, string> = {
  "/dashboard": "nav-dashboard",
  "/analysis":  "nav-analysis",
  "/compare":   "nav-compare",
  "/saved":     "nav-saved",
  "/reports":   "nav-reports",
  "/profile":   "nav-profile",
};

export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <nav className="flex shrink-0 flex-col gap-1.5 px-4" data-tour="sidebar-nav">
      {navItems.map((item) => {
        const active =
          item.href === "/dashboard"
            ? pathname === "/dashboard" || pathname === "/chat"
            : pathname === item.href || pathname.startsWith(`${item.href}/`);

        return (
          <Tooltip key={item.href}>
            <TooltipTrigger asChild>
              <Link
                href={item.href}
                onClick={onNavigate}
                data-tour={NAV_TOUR_IDS[item.href]}
                className={cn(
                  "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200",
                  active
                    ? "bg-brand-gradient shadow-brand-glow text-primary-foreground"
                    : "text-sidebar-foreground/65 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                )}
              >
                <item.icon className="size-4 shrink-0" />
                <span className="truncate">{item.title}</span>
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right" className="max-w-[220px]">
              {NAV_TOOLTIPS[item.href] ?? item.title}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </nav>
  );
}

