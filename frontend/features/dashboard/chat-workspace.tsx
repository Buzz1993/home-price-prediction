"use client";

// AI Chat Workspace: the main conversation column. Shows the message history
// (with structured result panels), an empty-state with suggested prompts, a
// loading indicator and an error state, plus the message composer pinned to the
// bottom. Reproduces the Streamlit chat-first workflow.

import { useEffect, useRef } from "react";
import { Bot, TriangleAlert } from "lucide-react";

import { ChatMessage } from "./chat-message";
import { ChatInput } from "./chat-input";
import { SuggestedPrompts } from "./suggested-prompts";
import { useWorkspace } from "./workspace-provider";

export function ChatWorkspace() {
  const { messages, isSending, error } = useWorkspace();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the latest message in view as the conversation grows.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 space-y-6 overflow-y-auto p-4">
        {messages.length === 0 && !isSending ? (
          <SuggestedPrompts />
        ) : (
          messages.map((message, i) => (
            <ChatMessage key={i} message={message} />
          ))
        )}

        {isSending && (
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <div className="flex size-8 items-center justify-center rounded-full bg-muted">
              <Bot className="size-4" />
            </div>
            <span className="animate-pulse">Analyzing your request…</span>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
            <TriangleAlert className="size-4 shrink-0" />
            <span>
              Something went wrong. Please make sure the backend is running and
              try again.
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t p-4">
        <ChatInput />
      </div>
    </div>
  );
}
