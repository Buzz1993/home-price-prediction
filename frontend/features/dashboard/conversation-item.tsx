"use client";

// A single conversation row in the ChatGPT-style sidebar (Phase 15.13). Clicking
// it switches to that complete workspace. The trailing menu exposes the
// conversation actions (Rename / Pin or Unpin / Delete); Rename switches the row
// to an inline input. Presentational only — all state lives in the workspace
// provider.

import { useEffect, useRef, useState } from "react";
import {
  Brain,
  Building2,
  Gauge,
  KeyRound,
  LineChart,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Scale,
  ShieldAlert,
  Trash2,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { Conversation } from "./conversations";

type ConversationItemProps = {
  conversation: Conversation;
  active: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (id: string) => void;
  onDelete: (id: string) => void;
};

// Pick a row icon from the conversation title so known conversation types
// (comparison, prediction, rental, …) are visually distinguishable. Purely
// cosmetic — conversations store no type field, so this infers from the
// title and falls back to a generic chat icon.
const TITLE_ICONS: Array<[RegExp, LucideIcon]> = [
  [/compar/i, Scale],
  [/predict|price/i, TrendingUp],
  [/invest|advis/i, Brain],
  [/rent/i, KeyRound],
  [/valuat/i, Gauge],
  [/risk/i, ShieldAlert],
  [/growth|future/i, LineChart],
  [/bhk|flat|apartment|property|home|house|luxur/i, Building2],
];

function iconForTitle(title: string): LucideIcon {
  for (const [pattern, icon] of TITLE_ICONS) {
    if (pattern.test(title)) return icon;
  }
  return MessageSquare;
}

// Compact relative timestamp for the metadata line ("Yesterday", "2 days
// ago", "Last week"). Returns null when the timestamp is missing so the line
// is only shown when data exists.
function relativeTime(timestamp: number): string | null {
  if (!timestamp) return null;

  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);

  if (timestamp >= startOfToday.getTime()) {
    const minutes = Math.floor((Date.now() - timestamp) / 60_000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes} min ago`;
    return "Today";
  }

  const dayMs = 86_400_000;
  const daysAgo = Math.ceil((startOfToday.getTime() - timestamp) / dayMs);
  if (daysAgo <= 1) return "Yesterday";
  if (daysAgo < 7) return `${daysAgo} days ago`;
  if (daysAgo < 14) return "Last week";
  return new Date(timestamp).toLocaleDateString();
}

export function ConversationItem({
  conversation,
  active,
  onSelect,
  onRename,
  onTogglePin,
  onDelete,
}: ConversationItemProps) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(conversation.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (renaming) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [renaming]);

  const RowIcon = iconForTitle(conversation.title);
  const updatedLabel = relativeTime(conversation.updatedAt);

  const startRename = () => {
    setDraft(conversation.title);
    setRenaming(true);
  };

  const commitRename = () => {
    setRenaming(false);
    const next = draft.trim();
    if (next && next !== conversation.title) onRename(conversation.id, next);
  };

  if (renaming) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commitRename}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commitRename();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setRenaming(false);
          }
        }}
        className="w-full rounded-lg border border-primary bg-white/10 px-2.5 py-2 text-sm text-sidebar-foreground outline-none ring-2 ring-primary/40"
      />
    );
  }

  return (
    <div
      className={cn(
        "group/item relative flex items-center gap-1 rounded-lg pr-1 transition-all duration-150",
        active
          ? "bg-sidebar-accent text-sidebar-foreground"
          : "hover:bg-white/5 hover:translate-x-0.5"
      )}
    >
      {/* Active accent bar on the left edge (premium ChatGPT-style cue). */}
      {active && (
        <span
          aria-hidden
          className="bg-brand-secondary absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full"
        />
      )}
      <button
        type="button"
        onClick={() => onSelect(conversation.id)}
        className="flex min-w-0 flex-1 items-start gap-2 px-2.5 py-2 text-left text-sm"
      >
        <RowIcon
          className={cn(
            "mt-0.5 size-3.5 shrink-0",
            active ? "text-brand-accent" : "text-sidebar-foreground/50"
          )}
        />
        <span className="min-w-0 flex-1">
          <span
            className={cn(
              "block truncate",
              active
                ? "font-medium text-sidebar-foreground"
                : "text-sidebar-foreground/75"
            )}
          >
            {conversation.title}
          </span>
          {updatedLabel && (
            <span className="block truncate text-[0.7rem] text-sidebar-foreground/45">
              {updatedLabel}
            </span>
          )}
        </span>
      </button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Conversation actions"
            className={cn(
              "shrink-0 text-sidebar-foreground/60 opacity-0 transition-opacity hover:bg-white/10 hover:text-sidebar-foreground group-hover/item:opacity-100 focus-visible:opacity-100 aria-expanded:bg-white/10 aria-expanded:text-sidebar-foreground aria-expanded:opacity-100",
              active && "opacity-100"
            )}
            onClick={(e) => e.stopPropagation()}
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={startRename}>
            <Pencil /> Rename
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onTogglePin(conversation.id)}>
            {conversation.pinned ? (
              <>
                <PinOff /> Unpin Chat
              </>
            ) : (
              <>
                <Pin /> Pin Chat
              </>
            )}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            onSelect={() => onDelete(conversation.id)}
          >
            <Trash2 /> Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
