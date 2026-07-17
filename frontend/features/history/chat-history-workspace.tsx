"use client";

// Chat History page body (Phase 18.9). A dedicated home for the AI
// conversation history that previously lived on the Profile page. Lists every
// stored conversation from the SAME workspace provider the Copilot uses
// (localStorage-persisted, Phase 15.13) — newest first, pinned separated —
// with its title, last-activity time, message and property counts. Opening a
// conversation switches the shared workspace to it and resumes it in the
// Copilot; Delete reuses the provider's existing action. No backend calls, no
// conversation logic — only relocated UI over the existing store.

import { useRouter } from "next/navigation";
import { History, MessageSquare, Pin, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import type { Conversation } from "@/features/dashboard/conversations";

export function ChatHistoryWorkspace() {
  const { conversations, switchConversation, deleteConversation } =
    useWorkspace();
  const router = useRouter();

  // Only conversations with actual messages; a fresh empty workspace is noise.
  const stored = conversations.filter((c) => c.messages.length > 0);
  const pinned = stored.filter((c) => c.pinned);
  const recent = stored
    .filter((c) => !c.pinned)
    .sort((a, b) => b.updatedAt - a.updatedAt);

  const openConversation = (id: string) => {
    switchConversation(id);
    router.push("/dashboard");
  };

  return (
    <div className="space-y-5">
      {/* Premium page hero — consistent with the other pages (Phase 18.7). */}
      <header className="space-y-3">
        <div className="flex items-center gap-4">
          <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 via-accent to-transparent ring-1 ring-primary/10">
            <History className="size-7 text-primary" />
          </div>
          <div className="space-y-1">
            <h1 className="font-heading text-3xl font-semibold tracking-tight">
              Chat History
            </h1>
            <p className="text-sm text-muted-foreground">
              Revisit and resume your previous AI conversations
            </p>
          </div>
        </div>
      </header>

      {stored.length === 0 ? (
        <div className="flex min-h-[50vh] items-center justify-center rounded-xl border border-dashed bg-gradient-to-br from-muted/20 to-transparent">
          <EmptyState
            icon={MessageSquare}
            title="No conversations yet"
            description="Start chatting with the EstateMind Copilot from the Dashboard, then find your conversations here."
            action={
              <Button onClick={() => router.push("/dashboard")}>
                Start a Conversation
              </Button>
            }
          />
        </div>
      ) : (
        <div className="mx-auto max-w-4xl space-y-6">
          {pinned.length > 0 && (
            <ConversationGroup
              title="Pinned"
              conversations={pinned}
              onOpen={openConversation}
              onDelete={deleteConversation}
            />
          )}
          <ConversationGroup
            title="Recent"
            conversations={recent}
            onOpen={openConversation}
            onDelete={deleteConversation}
          />
        </div>
      )}
    </div>
  );
}

function ConversationGroup({
  title,
  conversations,
  onOpen,
  onDelete,
}: {
  title: string;
  conversations: Conversation[];
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  if (conversations.length === 0) return null;
  return (
    <section className="space-y-3">
      <h2 className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
        {title === "Pinned" && <Pin className="size-3.5" />}
        {title}
        <span className="font-normal">({conversations.length})</span>
      </h2>
      <ul className="space-y-3">
        {conversations.map((conversation) => (
          <ConversationCard
            key={conversation.id}
            conversation={conversation}
            onOpen={onOpen}
            onDelete={onDelete}
          />
        ))}
      </ul>
    </section>
  );
}

function ConversationCard({
  conversation,
  onOpen,
  onDelete,
}: {
  conversation: Conversation;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const updated = new Date(conversation.updatedAt);
  // The first user message doubles as a short preview under the title.
  const firstUserMessage = conversation.messages.find(
    (m) => m.role === "user"
  )?.text;

  return (
    <li className="group flex items-start gap-4 rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float">
      <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary transition-colors duration-200 group-hover:bg-primary/20">
        <MessageSquare className="size-6" />
      </div>
      <button
        type="button"
        onClick={() => onOpen(conversation.id)}
        className="min-w-0 flex-1 space-y-1.5 text-left"
      >
        <p className="truncate font-semibold leading-tight">
          {conversation.title}
        </p>
        {firstUserMessage && (
          <p className="line-clamp-1 text-sm text-muted-foreground">
            {firstUserMessage}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span>
            {updated.toLocaleDateString()}{" "}
            {updated.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
          <Badge variant="secondary" className="px-2 py-0">
            {conversation.messages.length}{" "}
            {conversation.messages.length === 1 ? "message" : "messages"}
          </Badge>
          {conversation.properties.length > 0 && (
            <Badge variant="secondary" className="px-2 py-0">
              {conversation.properties.length}{" "}
              {conversation.properties.length === 1
                ? "property"
                : "properties"}
            </Badge>
          )}
        </div>
      </button>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={`Delete conversation ${conversation.title}`}
        className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
        onClick={() => onDelete(conversation.id)}
      >
        <Trash2 />
      </Button>
    </li>
  );
}
