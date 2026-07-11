"use client";

// A single conversation row in the ChatGPT-style sidebar (Phase 15.13). Clicking
// it switches to that complete workspace. The trailing menu exposes the
// conversation actions (Rename / Pin or Unpin / Delete); Rename switches the row
// to an inline input. Presentational only — all state lives in the workspace
// provider.

import { useEffect, useRef, useState } from "react";
import { MoreHorizontal, Pencil, Pin, PinOff, Trash2 } from "lucide-react";

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
        className="w-full rounded-lg border border-primary bg-background px-2.5 py-2 text-sm outline-none ring-2 ring-primary/30"
      />
    );
  }

  return (
    <div
      className={cn(
        "group/item relative flex items-center gap-1 rounded-lg pr-1 transition-all duration-150",
        active
          ? "bg-primary/10 text-foreground"
          : "hover:bg-muted hover:translate-x-0.5"
      )}
    >
      {/* Active accent bar on the left edge (premium ChatGPT-style cue). */}
      {active && (
        <span
          aria-hidden
          className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full bg-primary"
        />
      )}
      <button
        type="button"
        onClick={() => onSelect(conversation.id)}
        className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left text-sm"
      >
        <span
          className={cn(
            "truncate",
            active ? "font-medium" : "text-foreground/80"
          )}
        >
          {conversation.title}
        </span>
      </button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Conversation actions"
            className={cn(
              "shrink-0 opacity-0 transition-opacity group-hover/item:opacity-100 focus-visible:opacity-100 aria-expanded:opacity-100",
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
