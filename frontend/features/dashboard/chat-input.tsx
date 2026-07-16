"use client";

// Message composer for the chat workspace. Enter sends, Shift+Enter inserts a
// newline. Mirrors the Streamlit chat_input. While a response is streaming, the
// send button becomes a Stop button that cancels the active stream (Phase 15.9).

import { useState } from "react";
import { SendHorizontal, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useWorkspace } from "./workspace-provider";

export function ChatInput() {
  const { sendMessage, stopStreaming, isSending } = useWorkspace();
  const [value, setValue] = useState("");

  const submit = () => {
    if (!value.trim() || isSending) return;
    sendMessage(value);
    setValue("");
  };

  return (
    // Premium composer (Phase 18.1): a single rounded surface with a soft
    // shadow that focuses with a green ring; the send CTA sits inside it.
    <div className="flex items-end gap-2 rounded-2xl border bg-card p-2 shadow-float transition-shadow focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/30">
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask anything about properties…"
        rows={1}
        className="max-h-40 min-h-10 resize-none border-0 bg-transparent shadow-none focus-visible:border-transparent focus-visible:ring-0"
      />
      {isSending ? (
        <Button
          size="icon"
          variant="secondary"
          aria-label="Stop generating"
          onClick={stopStreaming}
          className="rounded-xl"
        >
          <Square className="fill-current" />
        </Button>
      ) : (
        <Button
          size="icon"
          aria-label="Send message"
          disabled={!value.trim()}
          onClick={submit}
          className="rounded-xl"
        >
          <SendHorizontal />
        </Button>
      )}
    </div>
  );
}
