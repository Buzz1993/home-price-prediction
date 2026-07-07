// AI chat history list for the Profile page. Reuses the shared ChatMessage bubble
// so a stored conversation renders exactly like the live chat — no bubble markup
// is duplicated. Each history entry is mapped onto the ChatMessage shape.

import { ChatMessage } from "@/features/dashboard/chat-message";
import type { ChatHistoryEntry } from "@/types/profile";

export function ChatHistoryList({ entries }: { entries: ChatHistoryEntry[] }) {
  return (
    <div className="space-y-4">
      {entries.map((entry, index) => (
        <ChatMessage
          key={entry.id ?? index}
          message={{ role: entry.role ?? "assistant", text: entry.message }}
        />
      ))}
    </div>
  );
}
