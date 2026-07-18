"use client";

// Message composer for the chat workspace. Enter sends, Shift+Enter inserts a
// newline. Mirrors the Streamlit chat_input. While a response is streaming, the
// send button becomes a Stop button that cancels the active stream (Phase 15.9).

import { useState } from "react";
import { SendHorizontal, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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
    // Premium composer (Phase 18.3, emphasized 18.16, persistent 18.17): the
    // primary CTA of the chat — a taller, elevated rounded surface that ALWAYS
    // shows the brand-green border and soft green glow (not just on focus), with
    // a one-time attention pulse on load and a minimal focus enhancement; the
    // send CTA sits inside it.
    <div
      data-tour="chat-input"
      className="composer-attention flex cursor-text items-end gap-2 rounded-3xl border border-ring bg-card p-3 shadow-float ring-3 ring-ring/25 transition-[box-shadow] duration-250 hover:shadow-float-lg focus-within:ring-ring/35"
    >
      <Tooltip>
        <TooltipTrigger asChild>
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
            className="max-h-40 min-h-11 resize-none border-0 bg-transparent px-3.5 py-2.5 text-base leading-relaxed shadow-none placeholder:font-medium placeholder:text-muted-foreground focus-visible:border-transparent focus-visible:ring-0"
          />
        </TooltipTrigger>
        <TooltipContent className="max-w-[220px]">
          Ask anything about properties using natural language.
        </TooltipContent>
      </Tooltip>
      {isSending ? (
        <Button
          size="icon-lg"
          variant="secondary"
          aria-label="Stop generating"
          onClick={stopStreaming}
          className="mb-0.5 rounded-xl shadow-sm"
        >
          <Square className="fill-current" />
        </Button>
      ) : (
        <Button
          size="icon-lg"
          aria-label="Send message"
          disabled={!value.trim()}
          onClick={submit}
          className="mb-0.5 rounded-xl transition-all duration-200 hover:translate-y-0 hover:scale-[1.04] active:scale-100 [&_svg:not([class*='size-'])]:size-4.5"
        >
          <SendHorizontal />
        </Button>
      )}
    </div>
  );
}
