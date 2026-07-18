// Client-side conversation search (Phase 19.1). Pure helpers powering the
// ChatGPT-style "Search chats..." experience in the conversation sidebar.
// Searches the already-loaded conversations — titles, user prompts, AI
// responses and the human-readable strings inside structured payloads (so
// "gym" also matches amenities inside search results). No backend requests,
// no business logic — filtering only.

import type { ChatMessage } from "@/types/dashboard";
import type { Conversation } from "./conversations";

export type ChatSearchMatch = {
  conversation: Conversation;
  /** Index of the first matched message, or -1 when only the title matched. */
  messageIndex: number;
  /** Short fragment around the match (empty for title-only matches). */
  snippet: string;
};

// All searchable text carried by one message: the bubble text plus the
// readable strings inside structured payloads. Case-preserving — callers
// lowercase for matching so snippets keep the original casing.
function messageSearchText(message: ChatMessage): string {
  const parts: string[] = [message.text];
  const response = message.response;
  if (response) {
    if (response.type === "search_results") {
      if (response.ai_explanation) parts.push(response.ai_explanation);
      for (const p of response.content) {
        parts.push(
          [
            p.project_name,
            p.location,
            p.locality,
            p.city,
            p.bhk_type,
            p.amenities_mcp,
            p.why_recommended,
          ]
            .filter(Boolean)
            .join(" ")
        );
      }
    } else if (response.type === "comparison") {
      for (const row of [response.content.winner, ...response.content.rankings]) {
        if (row) parts.push([row.verdict, row.comparison_reason].filter(Boolean).join(" "));
      }
    } else if (Array.isArray(response.content)) {
      // Rental / prediction / valuation / negotiation / advisor rows — index
      // every string cell so analysis wording is searchable too.
      for (const row of response.content) {
        parts.push(
          Object.values(row)
            .filter((v): v is string => typeof v === "string")
            .join(" ")
        );
      }
    }
  }
  return parts.join(" ");
}

// A compact fragment around the first occurrence of `query` (both lowercase
// handled by the caller passing the original text) for the result row.
function makeSnippet(text: string, query: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  const idx = clean.toLowerCase().indexOf(query);
  if (idx < 0) return "";
  const start = Math.max(0, idx - 24);
  const end = Math.min(clean.length, idx + query.length + 48);
  return (
    (start > 0 ? "…" : "") + clean.slice(start, end).trim() + (end < clean.length ? "…" : "")
  );
}

// Case-insensitive, partial-word search across titles, user prompts and AI
// responses. Returns one match per conversation (its first matching message),
// most recently updated first.
export function searchConversations(
  conversations: Conversation[],
  query: string
): ChatSearchMatch[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const matches: ChatSearchMatch[] = [];
  for (const conversation of conversations) {
    const titleMatch = conversation.title.toLowerCase().includes(q);

    let messageIndex = -1;
    let snippet = "";
    for (let i = 0; i < conversation.messages.length; i++) {
      const text = messageSearchText(conversation.messages[i]);
      if (text.toLowerCase().includes(q)) {
        messageIndex = i;
        snippet = makeSnippet(text, q);
        break;
      }
    }

    if (titleMatch || messageIndex >= 0) {
      matches.push({ conversation, messageIndex, snippet });
    }
  }
  return matches.sort(
    (a, b) => b.conversation.updatedAt - a.conversation.updatedAt
  );
}
