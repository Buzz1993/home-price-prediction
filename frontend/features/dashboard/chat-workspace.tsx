"use client";

// AI Chat Workspace: the main conversation column. Shows the message history
// (with structured result panels), an empty-state with suggested prompts, a
// thinking indicator, streamed assistant messages and an error state, plus the
// message composer pinned to the bottom. Reproduces the Streamlit chat-first
// workflow.

import { useEffect, useRef } from "react";
import { Bot } from "lucide-react";

import { ErrorState } from "@/components/ui/error-state";
import { ChatMessage } from "./chat-message";
import { ChatInput } from "./chat-input";
import { SuggestedPrompts } from "./suggested-prompts";
import { useWorkspace } from "./workspace-provider";

export function ChatWorkspace() {
  const { messages, isSending, phase, error, retryLastMessage } = useWorkspace();
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Whether to keep pinning to the newest message. Stays true while the user is
  // near the bottom; a manual scroll upward pauses auto-scroll (Phase 15.9) so
  // reading history is not interrupted, and returning to the bottom resumes it.
  const stickToBottom = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottom.current = distanceFromBottom < 80;
  };

  // Follow new content (including streamed tokens) only when the user is at the
  // bottom. Use instant scrolling while streaming so it tracks tokens smoothly.
  useEffect(() => {
    if (!stickToBottom.current) return;
    bottomRef.current?.scrollIntoView({
      behavior: phase === "streaming" ? "auto" : "smooth",
    });
  }, [messages, phase]);

  // Index of the newest search-results message. Only that message renders the
  // accumulated Property Results panel, so cards don't repeat per search.
  let lastSearchIndex = -1;
  messages.forEach((m, i) => {
    if (m.response?.type === "search_results") lastSearchIndex = i;
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 space-y-6 overflow-y-auto p-4"
      >
        {messages.length === 0 && !isSending ? (
          <SuggestedPrompts />
        ) : (
          messages.map((message, i) => (
            <ChatMessage
              key={i}
              message={message}
              // Suggestions (Phase 15.11) show only under the latest assistant
              // reply, and only once the response is complete (idle).
              showSuggestions={
                i === messages.length - 1 &&
                message.role === "assistant" &&
                !isSending
              }
              // The accumulated Property Results panel renders once, under the
              // newest search-results message (Phase 15.13).
              isLatestSearch={i === lastSearchIndex}
            />
          ))
        )}

        {/* Thinking indicator: shown while the backend runs, before the first
            streamed token replaces it with the live assistant message. */}
        {phase === "thinking" && (
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <div className="flex size-8 items-center justify-center rounded-full bg-muted">
              <Bot className="size-4" />
            </div>
            <span className="animate-pulse">Analyzing your request…</span>
          </div>
        )}

        {error && (
          <ErrorState
            title="Something went wrong"
            description="We couldn't process your last message. Please try again."
            onRetry={retryLastMessage}
            retrying={isSending}
          />
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t p-4">
        <ChatInput />
      </div>
    </div>
  );
}
