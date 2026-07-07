import {
  Bookmark,
  FileText,
  LayoutDashboard,
  MessageSquare,
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
// Routes point to the pages implemented in later phases.
export const navItems: NavItem[] = [
  { title: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { title: "AI Chat", href: "/chat", icon: MessageSquare },
  { title: "Property Comparison", href: "/compare", icon: Scale },
  { title: "AI Analysis", href: "/analysis", icon: Sparkles },
  { title: "Saved Properties", href: "/saved", icon: Bookmark },
  { title: "Reports", href: "/reports", icon: FileText },
  { title: "Profile", href: "/profile", icon: User },
];
