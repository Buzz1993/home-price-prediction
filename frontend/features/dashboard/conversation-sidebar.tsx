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
// restored on selection. "Search chats" (Phase 19.1) is a ChatGPT-style
// client-side search over the already-loaded conversations — titles, user
// prompts and AI responses — debounced ~200ms, with match highlighting and
// scroll-to-message on selection. No conversation, accumulation or backend
// logic is touched. The header, navigation and user footer are the same
// shared components the global sidebar uses, so nothing is duplicated.

import { useEffect, useState } from "react";
import { MessageSquare, Plus, Search, SearchX } from "lucide-react";

import { SidebarHeader } from "@/components/layout/sidebar-header";
import { SidebarNav } from "@/components/layout/sidebar-nav";
import { SidebarUser } from "@/components/layout/sidebar-user";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { searchConversations, type ChatSearchMatch } from "./chat-search";
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
    requestScrollToMessage,
  } = useWorkspace();
  const [query, setQuery] = useState("");
  // Debounced copy of the query (~200ms) so filtering doesn't run on every
  // keystroke of a fast typist while still feeling instantaneous.
  const [debounced, setDebounced] = useState("");

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(query), 200);
    return () => window.clearTimeout(t);
  }, [query]);

  const q = debounced.trim().toLowerCase();
  const searching = q.length > 0;

  // Default (no query) lists: Pinned first, then Recent — each ordered by most
  // recently updated. While searching, the list is replaced by search results.
  const byRecent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const pinned = byRecent.filter((c) => c.pinned);
  const recent = byRecent.filter((c) => !c.pinned);

  // ChatGPT-style search (Phase 19.1) over titles, user prompts and AI
  // responses — client-side only, over the already-loaded conversations.
  const results = searching ? searchConversations(conversations, q) : [];

  const handleSelect = (id: string) => {
    switchConversation(id);
    onNavigate?.();
  };

  // Open a search result: switch to the conversation and scroll to (and
  // briefly highlight) the matched message. Title-only matches just open.
  const handleResultSelect = (match: ChatSearchMatch) => {
    if (match.messageIndex >= 0) {
      requestScrollToMessage(match.conversation.id, match.messageIndex);
      onNavigate?.();
    } else {
      handleSelect(match.conversation.id);
    }
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
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className="w-full rounded-xl shadow-sm transition-all duration-200 hover:-translate-y-px hover:shadow-md active:translate-y-0"
              onClick={() => {
                newChat();
                onNavigate?.();
              }}
            >
              <Plus /> New Chat
            </Button>
          </TooltipTrigger>
          <TooltipContent className="max-w-[220px]">
            Start a new AI conversation.
          </TooltipContent>
        </Tooltip>
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

      {/* 6. Conversation list — the ONLY scroller in this sidebar. While
          searching it becomes the live search-results list (Phase 19.1). */}
      <nav
        data-tour="recent-chats"
        className="mt-5 min-h-0 flex-1 space-y-5 overflow-y-auto px-4 pb-3"
      >
        {searching ? (
          <Section title="Search Results">
            {results.length > 0 ? (
              <div className="space-y-0.5">
                {results.map((match) => (
                  <SearchResultItem
                    key={match.conversation.id}
                    match={match}
                    query={q}
                    active={match.conversation.id === activeId}
                    onSelect={handleResultSelect}
                  />
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 px-2.5 py-8 text-center">
                <span className="flex size-10 items-center justify-center rounded-xl bg-white/5 ring-1 ring-white/10">
                  <SearchX className="size-5 text-sidebar-foreground/50" />
                </span>
                <p className="text-xs text-sidebar-foreground/50">
                  No matching conversations found.
                </p>
              </div>
            )}
          </Section>
        ) : (
          <>
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
                  No recent conversations.
                </p>
              )}
            </Section>
          </>
        )}
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

// Wraps every occurrence of `query` (case-insensitive) in a highlight mark.
function Highlighted({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>;
  const parts: React.ReactNode[] = [];
  const lower = text.toLowerCase();
  let pos = 0;
  let idx = lower.indexOf(query);
  while (idx >= 0) {
    if (idx > pos) parts.push(text.slice(pos, idx));
    parts.push(
      <mark
        key={idx}
        className="text-brand-accent rounded-sm bg-primary/25 px-0.5 font-medium"
      >
        {text.slice(idx, idx + query.length)}
      </mark>
    );
    pos = idx + query.length;
    idx = lower.indexOf(query, pos);
  }
  if (pos < text.length) parts.push(text.slice(pos));
  return <>{parts}</>;
}

// One chat-search result row (Phase 19.1): highlighted title plus a snippet of
// the matched message when the match came from the conversation content.
function SearchResultItem({
  match,
  query,
  active,
  onSelect,
}: {
  match: ChatSearchMatch;
  query: string;
  active: boolean;
  onSelect: (match: ChatSearchMatch) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(match)}
      className={cn(
        "flex w-full items-start gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-all duration-150",
        active
          ? "bg-sidebar-accent text-sidebar-foreground"
          : "hover:bg-white/5"
      )}
    >
      <MessageSquare className="mt-0.5 size-3.5 shrink-0 text-sidebar-foreground/50" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sidebar-foreground/85">
          <Highlighted text={match.conversation.title} query={query} />
        </span>
        {match.snippet && (
          <span className="mt-0.5 line-clamp-2 block text-[0.7rem] leading-snug text-sidebar-foreground/50">
            <Highlighted text={match.snippet} query={query} />
          </span>
        )}
      </span>
    </button>
  );
}
