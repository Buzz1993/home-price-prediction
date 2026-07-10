"use client";

// Central workspace state for the dashboard (React Context per the project's
// state rules). Mirrors the Streamlit session_state used by the copilot:
//   chat_history            -> messages
//   comparison_tray         -> tray (property ids staged from search)
//   active_comparison_selection -> selected (ids ticked for comparison)
// The chat send lives here too so every panel shares one source of truth. All
// property logic stays in the backend; this only tracks UI state.
//
// Phase 15.9: chat responses stream over Server-Sent Events. Claude's search
// explanation arrives token-by-token into the active assistant message; the
// final structured payload renders exactly as before. Streaming only changes
// delivery — the backend workflow and every response shape are unchanged.

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

import { streamChatMessage } from "@/services/chat-service";
import type {
  ChatMessage,
  ChatRequest,
  ChatResponse,
} from "@/types/dashboard";

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
  if (response.type === "text") {
    return { role: "assistant", text: response.content };
  }
  return {
    role: "assistant",
    text: RESPONSE_TITLES[response.type],
    response,
  };
}

// A session-scoped id for the backend's conversational memory (Phase 15.7). It
// lives only for the lifetime of this provider (the active chat session): a
// fresh id is minted on mount and again after a new chat, so signing out and
// navigating away (which unmounts the provider) abandons the old memory. Uses
// crypto.randomUUID when available with a lightweight fallback.
function createSessionId(): string {
  const cryptoRef =
    typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
  if (cryptoRef?.randomUUID) return cryptoRef.randomUUID();
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// Chat lifecycle:
//   idle      -> nothing in flight
//   thinking  -> backend is running (search / analysis); no tokens yet
//   streaming -> Claude's explanation is arriving token-by-token
type ChatPhase = "idle" | "thinking" | "streaming";

type WorkspaceContextValue = {
  messages: ChatMessage[];
  tray: string[];
  selected: string[];
  // True while a response is in flight (thinking or streaming). Kept for
  // existing consumers (composer disabling, loading UI).
  isSending: boolean;
  // Finer-grained state for the streaming UI (thinking indicator vs cursor).
  phase: ChatPhase;
  isStreaming: boolean;
  error: Error | null;
  // Send a user message. `trayOverride` lets the tray's Compare action send the
  // active selection instead of the whole tray.
  sendMessage: (text: string, trayOverride?: string[]) => void;
  // Cancel the active streaming response cleanly (keeps any partial text).
  stopStreaming: () => void;
  // Re-run the last chat request after a failure. The user message stays in the
  // history, so no duplicate bubble is added. Undefined when there is nothing to
  // retry.
  retryLastMessage?: () => void;
  // Toggle a search result in/out of the tray.
  toggleTray: (id: string) => void;
  // Toggle whether a staged property is selected for comparison.
  toggleSelected: (id: string, checked: boolean) => void;
  removeFromTray: (id: string) => void;
  clearTray: () => void;
};

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tray, setTray] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [phase, setPhase] = useState<ChatPhase>("idle");
  const [error, setError] = useState<Error | null>(null);
  // Stable per-session id sent with every chat message so the backend can keep
  // conversational memory scoped to this session (Phase 15.7).
  const [sessionId] = useState(createSessionId);

  // The in-flight stream's abort controller and the last payload (for retry).
  const abortRef = useRef<AbortController | null>(null);
  const lastPayloadRef = useRef<ChatRequest | null>(null);

  // Append streamed tokens to the active assistant message (creating it on the
  // first token). Only the last message changes, keeping re-renders minimal.
  const appendDelta = useCallback((chunk: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.streaming) {
        return [...prev.slice(0, -1), { ...last, text: last.text + chunk }];
      }
      return [...prev, { role: "assistant", text: chunk, streaming: true }];
    });
  }, []);

  // Replace the streaming placeholder with the final structured message.
  const finalizeDone = useCallback((response: ChatResponse) => {
    const finalMessage = toAssistantMessage(response);
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.streaming) {
        return [...prev.slice(0, -1), finalMessage];
      }
      return [...prev, finalMessage];
    });
  }, []);

  // Stop the typing cursor on the active assistant message (on stop / error).
  // Clears the flag on EVERY streaming message — not just the last — so a
  // message superseded by a new send (which appends a user turn after it) also
  // has its cursor cleared. The `.some` guard keeps this a no-op (no re-render)
  // when nothing is streaming.
  const clearStreamingFlag = useCallback(() => {
    setMessages((prev) =>
      prev.some((m) => m.streaming)
        ? prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
        : prev
    );
  }, []);

  const runStream = useCallback(
    (payload: ChatRequest) => {
      // Cancel any previous stream and finalize its cursor before starting.
      abortRef.current?.abort();
      clearStreamingFlag();

      const controller = new AbortController();
      abortRef.current = controller;
      lastPayloadRef.current = payload;
      setError(null);
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
              appendDelta(event.text);
              break;
            case "done":
              finalizeDone(event.response);
              break;
            case "error":
              // Recoverable errors (e.g. the explanation was interrupted) are
              // followed by a `done` with any partial text, so they don't block
              // the response. Nothing to do here.
              break;
          }
        },
        controller.signal
      )
        .catch((err: unknown) => {
          // A cancellation (new send / Stop) is not an error.
          if (controller.signal.aborted) return;
          setError(err instanceof Error ? err : new Error("Chat request failed."));
          clearStreamingFlag();
        })
        .finally(() => {
          // Only the current stream resets shared state; a stream that was
          // superseded by a newer send leaves the newer one's state intact.
          if (abortRef.current === controller) {
            abortRef.current = null;
            clearStreamingFlag();
            setPhase("idle");
          }
        });
    },
    [appendDelta, finalizeDone, clearStreamingFlag]
  );

  const sendMessage = useCallback(
    (text: string, trayOverride?: string[]) => {
      const message = text.trim();
      if (!message) return;
      setMessages((prev) => [...prev, { role: "user", text: message }]);
      runStream({
        message,
        staged_property_ids: trayOverride ?? tray,
        session_id: sessionId,
      });
    },
    [runStream, tray, sessionId]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    clearStreamingFlag();
    setPhase("idle");
  }, [clearStreamingFlag]);

  // Re-send the exact request that failed (same message + tray). The user
  // message is already in the history, so drop the failed partial assistant
  // message before retrying to avoid a duplicate.
  const retryLastMessage = useCallback(() => {
    const payload = lastPayloadRef.current;
    if (!payload) return;
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant") return prev.slice(0, -1);
      return prev;
    });
    runStream(payload);
  }, [runStream]);

  const toggleTray = useCallback((id: string) => {
    setTray((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]
    );
    // Staging a property never auto-selects it for comparison (matches the
    // Streamlit tray: Compare stays unchecked until the user ticks it). Removing
    // a property from the tray also drops it from the selection.
    setSelected((prev) => prev.filter((p) => p !== id));
  }, []);

  const toggleSelected = useCallback((id: string, checked: boolean) => {
    setSelected((prev) => {
      if (checked) return prev.includes(id) ? prev : [...prev, id];
      return prev.filter((p) => p !== id);
    });
  }, []);

  const removeFromTray = useCallback((id: string) => {
    setTray((prev) => prev.filter((p) => p !== id));
    setSelected((prev) => prev.filter((p) => p !== id));
  }, []);

  const clearTray = useCallback(() => {
    setTray([]);
    setSelected([]);
  }, []);

  const value = useMemo<WorkspaceContextValue>(
    () => ({
      messages,
      tray,
      selected,
      isSending: phase !== "idle",
      phase,
      isStreaming: phase === "streaming",
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
      messages,
      tray,
      selected,
      phase,
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
