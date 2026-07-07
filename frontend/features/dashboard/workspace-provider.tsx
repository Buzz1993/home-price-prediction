"use client";

// Central workspace state for the dashboard (React Context per the project's
// state rules). Mirrors the Streamlit session_state used by the copilot:
//   chat_history            -> messages
//   comparison_tray         -> tray (property ids staged from search)
//   active_comparison_selection -> selected (ids ticked for comparison)
// The chat send mutation lives here too so every panel shares one source of
// truth. All property logic stays in the backend; this only tracks UI state.

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { useMutation } from "@tanstack/react-query";

import { sendChatMessage } from "@/services/chat-service";
import type { ChatMessage, ChatResponse } from "@/types/dashboard";

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

type WorkspaceContextValue = {
  messages: ChatMessage[];
  tray: string[];
  selected: string[];
  isSending: boolean;
  error: Error | null;
  // Send a user message. `trayOverride` lets the tray's Compare action send the
  // active selection instead of the whole tray.
  sendMessage: (text: string, trayOverride?: string[]) => void;
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

  const mutation = useMutation({
    mutationFn: sendChatMessage,
    onSuccess: (response) => {
      setMessages((prev) => [...prev, toAssistantMessage(response)]);
    },
  });

  const { mutate } = mutation;

  const sendMessage = useCallback(
    (text: string, trayOverride?: string[]) => {
      const message = text.trim();
      if (!message) return;
      setMessages((prev) => [...prev, { role: "user", text: message }]);
      mutate({ message, staged_property_ids: trayOverride ?? tray });
    },
    [mutate, tray]
  );

  // Re-send the exact request that failed (same message + tray) using the
  // mutation's last variables — available only after a send has been attempted.
  const lastVariables = mutation.variables;
  const retryLastMessage = useCallback(() => {
    if (lastVariables) mutate(lastVariables);
  }, [mutate, lastVariables]);

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
      isSending: mutation.isPending,
      error: mutation.error,
      sendMessage,
      retryLastMessage: mutation.isError ? retryLastMessage : undefined,
      toggleTray,
      toggleSelected,
      removeFromTray,
      clearTray,
    }),
    [
      messages,
      tray,
      selected,
      mutation.isPending,
      mutation.error,
      mutation.isError,
      sendMessage,
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
