// Chat API call. Thin wrapper over the documented POST /chat endpoint — the
// backend performs intent extraction, search, ranking and all analysis. See
// project_docs/03_API.md.

import { apiRequest } from "@/lib/api-client";
import type { ChatRequest, ChatResponse } from "@/types/dashboard";

export function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/chat", {
    method: "POST",
    body: payload,
  });
}
