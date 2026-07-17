// AI chat history list for the Profile page. Reuses the shared ChatMessage bubble
// so a stored conversation renders exactly like the live chat — no bubble markup
// is duplicated. Each history entry is mapped onto the ChatMessage shape.
//
// Phase 18.7: improved presentation with better spacing and visual hierarchy.

import { ChatMessage } from "@/features/dashboard/chat-message";
import type { ChatHistoryEntry } from "@/types/profile";

export function ChatHistoryList({ entries }: { entries: ChatHistoryEntry[] }) {
  return (
    <div className="space-y-4 rounded-xl border bg-gradient-to-br from-muted/20 to-transparent p-4">
      {entries.map((entry, index) => (
        <div key={entry.id ?? index} className="rounded-lg">
          <ChatMessage
            message={{ role: entry.role ?? "assistant", text: entry.message }}
          />
        </div>
      ))}
    </div>
  );
}
