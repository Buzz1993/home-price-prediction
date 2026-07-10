// Chat API call. Thin wrapper over the documented POST /chat endpoint — the
// backend performs intent extraction, search, ranking and all analysis. See
// project_docs/03_API.md.

import { API_BASE_URL, ApiError, apiRequest } from "@/lib/api-client";
import type {
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
} from "@/types/dashboard";

export function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/chat", {
    method: "POST",
    body: payload,
  });
}

// Streaming variant (Phase 15.9). Consumes the Server-Sent Events emitted by
// POST /chat/stream and forwards each event to `onEvent`. Streaming only changes
// delivery — the terminal `done` event carries the same ChatResponse envelope as
// sendChatMessage. Honors an AbortSignal so an active stream can be cancelled
// cleanly (a new send or a Stop action). All business logic stays in the backend.
export async function streamChatMessage(
  payload: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    // Backend failed before streaming started — surface a normal error so the
    // caller can show a friendly message and offer retry.
    let message = response.statusText || "Request failed";
    try {
      const data = await response.json();
      if (typeof data?.detail === "string") message = data.detail;
    } catch {
      // Non-JSON body — keep the status text.
    }
    throw new ApiError(message, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // SSE frames are separated by a blank line; each frame's `data:` lines hold
  // one JSON event. Parse incrementally as chunks arrive.
  const flush = (raw: string) => {
    const data = raw
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("");
    if (!data) return;
    try {
      onEvent(JSON.parse(data) as ChatStreamEvent);
    } catch {
      // Ignore malformed frames rather than crashing the chat.
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      flush(buffer.slice(0, separator));
      buffer = buffer.slice(separator + 2);
      separator = buffer.indexOf("\n\n");
    }
  }

  // Emit any trailing frame that arrived without a final blank line.
  if (buffer.trim()) flush(buffer);
}
