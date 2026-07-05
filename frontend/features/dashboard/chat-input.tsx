"use client";

// Message composer for the chat workspace. Enter sends, Shift+Enter inserts a
// newline. Mirrors the Streamlit chat_input.

import { useState } from "react";
import { SendHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useWorkspace } from "./workspace-provider";

export function ChatInput() {
  const { sendMessage, isSending } = useWorkspace();
  const [value, setValue] = useState("");

  const submit = () => {
    if (!value.trim() || isSending) return;
    sendMessage(value);
    setValue("");
  };

  return (
    <div className="flex items-end gap-2">
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
        className="max-h-40 min-h-10 resize-none"
      />
      <Button
        size="icon"
        aria-label="Send message"
        disabled={!value.trim() || isSending}
        onClick={submit}
      >
        <SendHorizontal />
      </Button>
    </div>
  );
}
