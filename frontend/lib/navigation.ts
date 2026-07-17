import {
  FileText,
  Heart,
  History,
  LayoutDashboard,
  Scale,
  Sparkles,
  User,
  type LucideIcon,
} from "lucide-react";

export type NavItem = {
  title: string;
  href: string;
  icon: LucideIcon;
};

// Primary navigation shared by the desktop sidebar and the mobile menu.
// The Dashboard is the single entry point for the Copilot Workspace — the
// former "AI Chat" item was removed (Phase 15.14) because it rendered the exact
// same workspace, creating duplicate navigation. The /chat route still works
// internally (see DashboardLayout WORKSPACE_ROUTES) for compatibility.
// "Property Comparison" (Phase 17.0) is the dedicated side-by-side comparison
// workspace at /compare; the quick Compare card inside AI Analysis remains for
// in-flow comparisons — both reuse the same backend comparison endpoint.
// "Chat History" (Phase 18.9) lists the stored Copilot conversations — the
// history UI that previously lived on the Profile page.
export const navItems: NavItem[] = [
  { title: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { title: "AI Analysis", href: "/analysis", icon: Sparkles },
  { title: "Chat History", href: "/history", icon: History },
  { title: "Property Comparison", href: "/compare", icon: Scale },
  { title: "Saved Properties", href: "/saved", icon: Heart },
  { title: "Reports", href: "/reports", icon: FileText },
  { title: "Profile", href: "/profile", icon: User },
];
