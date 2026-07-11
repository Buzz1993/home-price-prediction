"use client";

// ChatGPT-style Conversation Sidebar (Phase 15.13). The far-left column of the
// Copilot workspace. Lists conversations in Pinned / Recent sections, starts a
// New Chat, and (per the chosen layout) folds the app's global navigation into
// the footer so the workspace can run full-bleed without a second sidebar. Each
// conversation is a complete EstateMind workspace restored on selection.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Plus } from "lucide-react";

import { Brand } from "@/components/layout/brand";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { navItems } from "@/lib/navigation";
import { ConversationItem } from "./conversation-item";
import { useWorkspace } from "./workspace-provider";

export function ConversationSidebar({
  className,
  onNavigate,
}: {
  className?: string;
  // Lets the mobile sheet close after picking a conversation or a nav link.
  onNavigate?: () => void;
}) {
  const {
    conversations,
    activeId,
    newChat,
    switchConversation,
    renameConversation,
    togglePin,
    deleteConversation,
  } = useWorkspace();
  const pathname = usePathname();

  // Pinned first, then Recent — each ordered by most recently updated.
  const byRecent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const pinned = byRecent.filter((c) => c.pinned);
  const recent = byRecent.filter((c) => !c.pinned);

  const handleSelect = (id: string) => {
    switchConversation(id);
    onNavigate?.();
  };

  return (
    <aside
      className={cn(
        "flex h-full w-full flex-col bg-sidebar text-sidebar-foreground",
        className
      )}
    >
      <div className="flex h-16 items-center border-b px-4">
        <Brand />
      </div>

      <div className="p-3">
        <Button
          className="w-full justify-center"
          onClick={() => {
            newChat();
            onNavigate?.();
          }}
        >
          <Plus /> New Chat
        </Button>
      </div>

      <nav className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 pb-3">
        {pinned.length > 0 && (
          <Section title="Pinned">
            {pinned.map((c) => (
              <ConversationItem
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                onSelect={handleSelect}
                onRename={renameConversation}
                onTogglePin={togglePin}
                onDelete={deleteConversation}
              />
            ))}
          </Section>
        )}

        <Section title="Recent">
          {recent.length > 0 ? (
            recent.map((c) => (
              <ConversationItem
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                onSelect={handleSelect}
                onRename={renameConversation}
                onTogglePin={togglePin}
                onDelete={deleteConversation}
              />
            ))
          ) : (
            <p className="px-2.5 py-2 text-xs text-muted-foreground">
              No recent conversations.
            </p>
          )}
        </Section>
      </nav>

      {/* Global navigation folded into the sidebar footer so the workspace runs
          full-bleed. Links to the rest of the app. */}
      <div className="border-t p-3">
        <div className="grid grid-cols-2 gap-1">
          {navItems.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onNavigate}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                <item.icon className="size-3.5 shrink-0" />
                <span className="truncate">{item.title}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <p className="px-2.5 text-[0.7rem] font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}
