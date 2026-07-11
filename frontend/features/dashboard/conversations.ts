// Conversation model + pure helpers for the Copilot workspace (Phase 15.13).
//
// Each conversation is one complete EstateMind workspace: chat messages, the
// accumulated (deduplicated) property collection discovered during the
// conversation, the evaluation tray, and the comparison selection. This is
// purely frontend UI state — no business logic. All property data still comes
// from the existing backend; this module only stores and organizes it.
//
// Persistence is client-only localStorage so conversations survive a reload and
// can be resumed (the ChatGPT-style Recent/Pinned lists). No backend, API
// contract or database is involved.

import type { ChatMessage, SearchResult } from "@/types/dashboard";

export type Conversation = {
  // Also used as the backend `session_id` (Phase 15.7 conversational memory), so
  // each conversation has its own server-side memory scope.
  id: string;
  title: string;
  pinned: boolean;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  // Every unique property discovered across all searches in this conversation.
  properties: SearchResult[];
  // Evaluation tray (staged property ids) and the comparison selection.
  tray: string[];
  selected: string[];
};

const STORAGE_KEY = "estatemind:workspace:v1";
// Keep the persisted history bounded.
const MAX_CONVERSATIONS = 50;

// A stable id used both as the conversation id and the backend session id. Uses
// crypto.randomUUID when available with a lightweight fallback.
export function createId(): string {
  const cryptoRef =
    typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
  if (cryptoRef?.randomUUID) return cryptoRef.randomUUID();
  return `id-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// A fresh, empty workspace.
export function createConversation(now: number): Conversation {
  return {
    id: createId(),
    title: "New chat",
    pinned: false,
    createdAt: now,
    updatedAt: now,
    messages: [],
    properties: [],
    tray: [],
    selected: [],
  };
}

// Accumulate newly discovered properties into the conversation collection,
// preserving order and dropping duplicates by property id — so a second search
// appends only the properties not already present (5 + 5 unique -> 10).
export function mergeProperties(
  existing: SearchResult[],
  incoming: SearchResult[]
): SearchResult[] {
  if (!incoming || incoming.length === 0) return existing;
  const seen = new Set(existing.map((p) => p.id));
  const additions = incoming.filter((p) => p && p.id && !seen.has(p.id));
  if (additions.length === 0) return existing;
  return [...existing, ...additions];
}

// Derive a short conversation title from the first user message.
export function deriveTitle(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, " ");
  if (!trimmed) return "New chat";
  return trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed;
}

export type WorkspaceSnapshot = {
  conversations: Conversation[];
  activeId: string | null;
};

// Load the persisted workspace (client only). Returns null when nothing is
// stored or the payload is unreadable, so the caller can seed a fresh workspace.
export function loadWorkspace(): WorkspaceSnapshot | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as WorkspaceSnapshot;
    if (!parsed || !Array.isArray(parsed.conversations)) return null;
    // Clear any stale streaming flags left by a previous session so a persisted
    // half-streamed message doesn't render a frozen typing cursor after reload.
    for (const c of parsed.conversations) {
      if (Array.isArray(c.messages)) {
        for (const m of c.messages) if (m.streaming) m.streaming = false;
      }
    }
    return parsed;
  } catch {
    return null;
  }
}

// Persist the workspace (client only), bounded to the most recent conversations.
export function saveWorkspace(snapshot: WorkspaceSnapshot): void {
  if (typeof window === "undefined") return;
  try {
    const payload: WorkspaceSnapshot = {
      activeId: snapshot.activeId,
      conversations: snapshot.conversations.slice(0, MAX_CONVERSATIONS),
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore quota / serialization errors — persistence is best-effort.
  }
}
