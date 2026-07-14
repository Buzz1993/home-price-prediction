import {
  Bookmark,
  FileText,
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
export const navItems: NavItem[] = [
  { title: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { title: "Property Comparison", href: "/compare", icon: Scale },
  { title: "AI Analysis", href: "/analysis", icon: Sparkles },
  { title: "Saved Properties", href: "/saved", icon: Bookmark },
  { title: "Reports", href: "/reports", icon: FileText },
  { title: "Profile", href: "/profile", icon: User },
];
