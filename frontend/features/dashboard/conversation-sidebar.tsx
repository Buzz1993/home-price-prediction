"use client";

// ChatGPT-style Conversation Sidebar (Phase 15.13). The far-left column of the
// Copilot workspace, laid out top→bottom as:
//
//   EstateMind logo → New Chat → Search Chats → Pinned → Recent → Global nav
//
// It is a bounded flex column: the logo, New Chat, search box and global
// navigation stay fixed, and ONLY the Pinned/Recent conversation list scrolls
// independently. Each conversation is a complete EstateMind workspace restored
// on selection. "Search Chats" filters the already-loaded conversation titles on
// the client — it does not touch the conversation, accumulation or backend logic.

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Plus, Search } from "lucide-react";

import { Brand } from "@/components/layout/brand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  const [query, setQuery] = useState("");

  // Pinned first, then Recent — each ordered by most recently updated, then
  // filtered by the "Search Chats" query (title match, client-side only).
  const q = query.trim().toLowerCase();
  const byRecent = [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .filter((c) => (q ? c.title.toLowerCase().includes(q) : true));
  const pinned = byRecent.filter((c) => c.pinned);
  const recent = byRecent.filter((c) => !c.pinned);

  const handleSelect = (id: string) => {
    switchConversation(id);
    onNavigate?.();
  };

  const renderItem = (id: string) => {
    const c = conversations.find((x) => x.id === id)!;
    return (
      <ConversationItem
        key={c.id}
        conversation={c}
        active={c.id === activeId}
        onSelect={handleSelect}
        onRename={renameConversation}
        onTogglePin={togglePin}
        onDelete={deleteConversation}
      />
    );
  };

  return (
    <aside
      className={cn(
        "flex h-full min-h-0 w-full flex-col bg-sidebar text-sidebar-foreground",
        className
      )}
    >
      {/* Brand — fixed. */}
      <div className="flex h-16 shrink-0 items-center border-b px-4">
        <Brand />
      </div>

      {/* New Chat + Search Chats — fixed. */}
      <div className="shrink-0 space-y-2 p-3">
        <Button
          className="w-full justify-center shadow-sm"
          onClick={() => {
            newChat();
            onNavigate?.();
          }}
        >
          <Plus /> New Chat
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats"
            aria-label="Search chats"
            className="h-9 bg-background/60 pl-8 text-sm"
          />
        </div>
      </div>

      {/* Conversation list — the ONLY scroller in this sidebar. */}
      <nav className="min-h-0 flex-1 space-y-5 overflow-y-auto px-3 pb-3">
        {pinned.length > 0 && (
          <Section title="Pinned">{pinned.map((c) => renderItem(c.id))}</Section>
        )}

        <Section title="Recent">
          {recent.length > 0 ? (
            recent.map((c) => renderItem(c.id))
          ) : (
            <p className="px-2.5 py-2 text-xs text-muted-foreground">
              {q ? "No chats match your search." : "No recent conversations."}
            </p>
          )}
        </Section>
      </nav>

      {/* Global navigation — fixed footer so the workspace runs full-bleed. */}
      <div className="shrink-0 border-t p-3">
        <p className="px-1 pb-1.5 text-[0.7rem] font-semibold uppercase tracking-wide text-muted-foreground">
          Navigate
        </p>
        <div className="grid grid-cols-2 gap-1">
          {navItems.map((item) => {
            // Dashboard is the single Copilot Workspace entry point, so it stays
            // highlighted anywhere inside the workspace — including the retained
            // /chat compatibility route which renders the same shell.
            const active =
              item.href === "/dashboard"
                ? pathname === "/dashboard" || pathname === "/chat"
                : pathname === item.href ||
                  pathname.startsWith(`${item.href}/`);
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
