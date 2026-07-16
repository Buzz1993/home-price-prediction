"use client";

// Dashboard conversation sidebar (Phase 15.13, redesigned as a modern AI
// SaaS rail). The user's chat workspace home, laid out top→bottom as:
//
//   Brand + badge → Navigation → New Chat → Search chats → Pinned/Recent →
//   Signed-in user
//
// It is a bounded flex column: header, nav, New Chat, search and the user
// footer stay fixed, and ONLY the Pinned/Recent conversation list scrolls
// independently. Each conversation is a complete EstateMind workspace
// restored on selection. "Search chats" filters the already-loaded titles on
// the client — no conversation, accumulation or backend logic is touched.
// The header, navigation and user footer are the same shared components the
// global sidebar uses, so nothing is duplicated.

import { useState } from "react";
import { Plus, Search } from "lucide-react";

import { SidebarHeader } from "@/components/layout/sidebar-header";
import { SidebarNav } from "@/components/layout/sidebar-nav";
import { SidebarUser } from "@/components/layout/sidebar-user";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
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
  const [query, setQuery] = useState("");

  // Pinned first, then Recent — each ordered by most recently updated, then
  // filtered by the search query (title match, client-side only).
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
        // Carries the dark purple rail gradient itself (Phase 18.2) so both
        // hosts — the workspace shell's floating panel and the mobile sheet —
        // render the same premium surface.
        "bg-sidebar-gradient flex h-full min-h-0 w-full flex-col text-sidebar-foreground",
        className
      )}
    >
      {/* 1. Brand + badge — fixed (shared header). */}
      <SidebarHeader />

      {/* 2. Primary navigation — fixed (shared nav rows). */}
      <SidebarNav onNavigate={onNavigate} />

      {/* 3–5. New Chat (primary CTA) + Search chats — fixed. */}
      <div className="shrink-0 space-y-2.5 px-4 pt-5">
        <Button
          className="w-full rounded-xl shadow-sm transition-all duration-200 hover:-translate-y-px hover:shadow-md active:translate-y-0"
          onClick={() => {
            newChat();
            onNavigate?.();
          }}
        >
          <Plus /> New Chat
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-sidebar-foreground/50" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats..."
            aria-label="Search chats"
            className="h-9 rounded-xl border-white/10 bg-white/5 pl-9 text-sm text-sidebar-foreground placeholder:text-sidebar-foreground/40"
          />
        </div>
      </div>

      {/* 6. Conversation list — the ONLY scroller in this sidebar. */}
      <nav className="mt-5 min-h-0 flex-1 space-y-5 overflow-y-auto px-4 pb-3">
        {pinned.length > 0 && (
          <Section title="Pinned">{pinned.map((c) => renderItem(c.id))}</Section>
        )}

        <Section title="Recent">
          {recent.length > 0 ? (
            <div className="space-y-0.5 duration-300 animate-in fade-in slide-in-from-left-1">
              {recent.map((c) => renderItem(c.id))}
            </div>
          ) : (
            <p className="px-2.5 py-2 text-xs text-sidebar-foreground/50">
              {q ? "No chats match your search." : "No recent conversations."}
            </p>
          )}
        </Section>
      </nav>

      {/* 7. Signed-in user — fixed footer (shared component). */}
      <SidebarUser onNavigate={onNavigate} />
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
      <p className="px-2.5 text-[0.7rem] font-semibold uppercase tracking-wider text-sidebar-foreground/50">
        {title}
      </p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}
