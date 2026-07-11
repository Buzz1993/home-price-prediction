"use client";

// Central workspace state for the Copilot (React Context per the project's state
// rules). Phase 15.13 turns the single implicit session into a multi-conversation
// workspace: each conversation is one complete EstateMind workspace (chat
// messages, an accumulated deduplicated property collection, the evaluation tray
// and the comparison selection). Switching conversations restores the full
// workspace; New Chat creates an empty one. Conversations persist to
// localStorage so they survive a reload (the Recent/Pinned lists).
//
// All property logic stays in the backend; this only tracks UI state. Chat still
// streams over Server-Sent Events (Phase 15.9) — the search explanation arrives
// token-by-token and the final structured payload renders exactly as before. On
// a search result the returned properties are accumulated into the active
// conversation (deduplicated by id), and BOTH the Property Results panel and the
// Interactive Property Map render from that single accumulated collection.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { streamChatMessage } from "@/services/chat-service";
import type {
  ChatMessage,
  ChatRequest,
  ChatResponse,
  SearchResult,
} from "@/types/dashboard";
import {
  Conversation,
  createConversation,
  deriveTitle,
  loadWorkspace,
  mergeProperties,
  saveWorkspace,
} from "./conversations";

// Header text shown above each structured assistant payload. Matches the
// RESPONSE_CONFIG titles in the Streamlit reference.
const RESPONSE_TITLES: Record<ChatResponse["type"], string> = {
  text: "",
  search_results:
    "Here are the highest ranking properties matching your intent:",
  comparison: "Investment analysis complete.",
  rental: "Rental performance summary:",
  prediction: "Predicted price forecast:",
  negotiation: "Negotiation strategy compiled:",
  valuation: "Valuation assessment complete:",
  advisor: "Investment advice compiled:",
};

function toAssistantMessage(response: ChatResponse): ChatMessage {
  // Follow-up suggestions (Phase 15.11) ride on the response envelope; surface
  // them on the message so the chat renderer can show quick-action chips.
  const suggestions = response.suggestions;
  if (response.type === "text") {
    return { role: "assistant", text: response.content, suggestions };
  }
  return {
    role: "assistant",
    text: RESPONSE_TITLES[response.type],
    response,
    suggestions,
  };
}

// Chat lifecycle:
//   idle      -> nothing in flight
//   thinking  -> backend is running (search / analysis); no tokens yet
//   streaming -> Claude's explanation is arriving token-by-token
type ChatPhase = "idle" | "thinking" | "streaming";

type WorkspaceContextValue = {
  // --- Active conversation (derived) -------------------------------------
  messages: ChatMessage[];
  // Accumulated, deduplicated properties for the active conversation. Both the
  // Property Results panel and the map render from this.
  properties: SearchResult[];
  tray: string[];
  selected: string[];
  // The property currently highlighted (card <-> marker sync). Ephemeral.
  selectedPropertyId: string | null;
  setSelectedPropertyId: (id: string | null) => void;

  // --- Conversation list --------------------------------------------------
  conversations: Conversation[];
  activeId: string | null;
  newChat: () => void;
  switchConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  deleteConversation: (id: string) => void;

  // --- Chat ---------------------------------------------------------------
  // True while a response is in flight for the ACTIVE conversation.
  isSending: boolean;
  phase: ChatPhase;
  isStreaming: boolean;
  error: Error | null;
  sendMessage: (text: string, trayOverride?: string[]) => void;
  stopStreaming: () => void;
  retryLastMessage?: () => void;

  // --- Evaluation tray ----------------------------------------------------
  toggleTray: (id: string) => void;
  toggleSelected: (id: string, checked: boolean) => void;
  removeFromTray: (id: string) => void;
  clearTray: () => void;
};

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const [phase, setPhase] = useState<ChatPhase>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [selectedPropertyId, setSelectedPropertyId] = useState<string | null>(
    null
  );
  // Which conversation the in-flight stream belongs to, so the thinking /
  // streaming indicator only shows on that conversation (switching away hides
  // it) while the stream keeps updating its own conversation.
  const [streamingConvId, setStreamingConvId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const lastPayloadRef = useRef<ChatRequest | null>(null);

  // Hydrate the persisted workspace once, on the client. Reading localStorage
  // must happen after mount to avoid an SSR hydration mismatch.
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    const stored = loadWorkspace();
    if (stored && stored.conversations.length > 0) {
      setConversations(stored.conversations);
      const validActive =
        stored.activeId &&
        stored.conversations.some((c) => c.id === stored.activeId)
          ? stored.activeId
          : stored.conversations[0].id;
      setActiveId(validActive);
    } else {
      const conv = createConversation(Date.now());
      setConversations([conv]);
      setActiveId(conv.id);
    }
    setHydrated(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  // Persist on every change once hydrated.
  useEffect(() => {
    if (!hydrated) return;
    saveWorkspace({ conversations, activeId });
  }, [conversations, activeId, hydrated]);

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId]
  );

  // Immutable update of a single conversation by id.
  const updateConversation = useCallback(
    (id: string, updater: (c: Conversation) => Conversation) => {
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? updater(c) : c))
      );
    },
    []
  );

  // --- Streaming helpers (scoped to a specific conversation id) -----------

  const appendDelta = useCallback(
    (convId: string, chunk: string) => {
      updateConversation(convId, (c) => {
        const last = c.messages[c.messages.length - 1];
        if (last && last.role === "assistant" && last.streaming) {
          return {
            ...c,
            messages: [
              ...c.messages.slice(0, -1),
              { ...last, text: last.text + chunk },
            ],
          };
        }
        return {
          ...c,
          messages: [
            ...c.messages,
            { role: "assistant", text: chunk, streaming: true },
          ],
        };
      });
    },
    [updateConversation]
  );

  // Replace the streaming placeholder with the final structured message and, for
  // a search result, accumulate the returned properties into the conversation.
  const finalizeDone = useCallback(
    (convId: string, response: ChatResponse) => {
      const finalMessage = toAssistantMessage(response);
      updateConversation(convId, (c) => {
        const last = c.messages[c.messages.length - 1];
        const messages =
          last && last.role === "assistant" && last.streaming
            ? [...c.messages.slice(0, -1), finalMessage]
            : [...c.messages, finalMessage];
        const properties =
          response.type === "search_results"
            ? mergeProperties(c.properties, response.content)
            : c.properties;
        return { ...c, messages, properties };
      });
    },
    [updateConversation]
  );

  // Stop the typing cursor on any streaming message of a conversation.
  const clearStreamingFlag = useCallback(
    (convId: string) => {
      updateConversation(convId, (c) =>
        c.messages.some((m) => m.streaming)
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.streaming ? { ...m, streaming: false } : m
              ),
            }
          : c
      );
    },
    [updateConversation]
  );

  const runStream = useCallback(
    (convId: string, payload: ChatRequest) => {
      // Cancel any previous stream and finalize its cursor before starting.
      abortRef.current?.abort();
      const prevConv = streamingConvId;
      if (prevConv) clearStreamingFlag(prevConv);

      const controller = new AbortController();
      abortRef.current = controller;
      lastPayloadRef.current = payload;
      setError(null);
      setStreamingConvId(convId);
      setPhase("thinking");

      streamChatMessage(
        payload,
        (event) => {
          if (controller.signal.aborted) return;
          switch (event.type) {
            case "thinking":
              setPhase("thinking");
              break;
            case "delta":
              setPhase("streaming");
              appendDelta(convId, event.text);
              break;
            case "done":
              finalizeDone(convId, event.response);
              break;
            case "error":
              // Recoverable errors are followed by a `done` with any partial
              // text, so they don't block the response. Nothing to do here.
              break;
          }
        },
        controller.signal
      )
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setError(
            err instanceof Error ? err : new Error("Chat request failed.")
          );
          clearStreamingFlag(convId);
        })
        .finally(() => {
          // Only the current stream resets shared state; a superseded stream
          // leaves the newer one's state intact.
          if (abortRef.current === controller) {
            abortRef.current = null;
            clearStreamingFlag(convId);
            setPhase("idle");
            setStreamingConvId(null);
          }
        });
    },
    [appendDelta, finalizeDone, clearStreamingFlag, streamingConvId]
  );

  const sendMessage = useCallback(
    (text: string, trayOverride?: string[]) => {
      const message = text.trim();
      if (!message || !activeId) return;
      const convId = activeId;
      const stagedIds = trayOverride ?? active?.tray ?? [];
      updateConversation(convId, (c) => ({
        ...c,
        title: c.messages.length === 0 ? deriveTitle(message) : c.title,
        updatedAt: Date.now(),
        messages: [...c.messages, { role: "user", text: message }],
      }));
      runStream(convId, {
        message,
        staged_property_ids: stagedIds,
        session_id: convId,
      });
    },
    [activeId, active, updateConversation, runStream]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (streamingConvId) clearStreamingFlag(streamingConvId);
    setPhase("idle");
    setStreamingConvId(null);
  }, [clearStreamingFlag, streamingConvId]);

  const retryLastMessage = useCallback(() => {
    const payload = lastPayloadRef.current;
    if (!payload) return;
    const convId = payload.session_id ?? activeId;
    if (!convId) return;
    updateConversation(convId, (c) => {
      const last = c.messages[c.messages.length - 1];
      if (last && last.role === "assistant") {
        return { ...c, messages: c.messages.slice(0, -1) };
      }
      return c;
    });
    runStream(convId, payload);
  }, [activeId, updateConversation, runStream]);

  // --- Conversation actions ----------------------------------------------

  const newChat = useCallback(() => {
    setError(null);
    setSelectedPropertyId(null);
    // If the active conversation is already an untouched blank workspace, just
    // stay on it instead of stacking empty conversations (ChatGPT behaviour).
    if (active && active.messages.length === 0) return;
    const conv = createConversation(Date.now());
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  }, [active]);

  const switchConversation = useCallback(
    (id: string) => {
      if (id === activeId) return;
      setError(null);
      setSelectedPropertyId(null);
      setActiveId(id);
    },
    [activeId]
  );

  const renameConversation = useCallback(
    (id: string, title: string) => {
      const next = title.trim();
      updateConversation(id, (c) => ({
        ...c,
        title: next || c.title,
      }));
    },
    [updateConversation]
  );

  const togglePin = useCallback(
    (id: string) => {
      updateConversation(id, (c) => ({ ...c, pinned: !c.pinned }));
    },
    [updateConversation]
  );

  const deleteConversation = useCallback(
    (id: string) => {
      // Handle active-id fixup here (a user action) rather than in an effect, so
      // deleting the active conversation immediately falls back to another one.
      const remaining = conversations.filter((c) => c.id !== id);
      if (remaining.length > 0) {
        setConversations(remaining);
        if (activeId === id) setActiveId(remaining[0].id);
      } else {
        const conv = createConversation(Date.now());
        setConversations([conv]);
        setActiveId(conv.id);
      }
      setSelectedPropertyId(null);
    },
    [conversations, activeId]
  );

  // --- Evaluation tray (scoped to the active conversation) ----------------

  const toggleTray = useCallback(
    (id: string) => {
      if (!activeId) return;
      updateConversation(activeId, (c) => {
        const inTray = c.tray.includes(id);
        return {
          ...c,
          tray: inTray ? c.tray.filter((p) => p !== id) : [...c.tray, id],
          // Staging never auto-selects for comparison; removing drops selection.
          selected: c.selected.filter((p) => p !== id),
        };
      });
    },
    [activeId, updateConversation]
  );

  const toggleSelected = useCallback(
    (id: string, checked: boolean) => {
      if (!activeId) return;
      updateConversation(activeId, (c) => {
        const selected = checked
          ? c.selected.includes(id)
            ? c.selected
            : [...c.selected, id]
          : c.selected.filter((p) => p !== id);
        return { ...c, selected };
      });
    },
    [activeId, updateConversation]
  );

  const removeFromTray = useCallback(
    (id: string) => {
      if (!activeId) return;
      updateConversation(activeId, (c) => ({
        ...c,
        tray: c.tray.filter((p) => p !== id),
        selected: c.selected.filter((p) => p !== id),
      }));
    },
    [activeId, updateConversation]
  );

  const clearTray = useCallback(() => {
    if (!activeId) return;
    updateConversation(activeId, (c) => ({ ...c, tray: [], selected: [] }));
  }, [activeId, updateConversation]);

  // The indicator only reflects the active conversation's stream.
  const activePhase: ChatPhase =
    streamingConvId === activeId ? phase : "idle";

  const value = useMemo<WorkspaceContextValue>(
    () => ({
      messages: active?.messages ?? [],
      properties: active?.properties ?? [],
      tray: active?.tray ?? [],
      selected: active?.selected ?? [],
      selectedPropertyId,
      setSelectedPropertyId,
      conversations,
      activeId,
      newChat,
      switchConversation,
      renameConversation,
      togglePin,
      deleteConversation,
      isSending: activePhase !== "idle",
      phase: activePhase,
      isStreaming: activePhase === "streaming",
      error,
      sendMessage,
      stopStreaming,
      retryLastMessage: error ? retryLastMessage : undefined,
      toggleTray,
      toggleSelected,
      removeFromTray,
      clearTray,
    }),
    [
      active,
      selectedPropertyId,
      conversations,
      activeId,
      newChat,
      switchConversation,
      renameConversation,
      togglePin,
      deleteConversation,
      activePhase,
      error,
      sendMessage,
      stopStreaming,
      retryLastMessage,
      toggleTray,
      toggleSelected,
      removeFromTray,
      clearTray,
    ]
  );

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): WorkspaceContextValue {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used within a WorkspaceProvider");
  }
  return context;
}
