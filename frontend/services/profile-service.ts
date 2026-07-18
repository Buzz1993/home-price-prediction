// Profile API calls. Thin wrappers over the documented Profile endpoints
// (GET /profile, GET /chat-history, GET /reports — see project_docs/03_API.md).
// The backend owns the user account, conversation history and report list; the
// frontend only reads and renders them.

import { apiRequest } from "@/lib/api-client";
import type { ChatHistoryEntry, ReportSummary, User } from "@/types/profile";

// GET /profile — retrieve the authenticated user's profile. The backend
// resolves the Bearer token to the account (Phase 18.18), so the token is
// required.
export function getProfile(token: string | null): Promise<User> {
  return apiRequest<User>("/profile", { token });
}

// GET /chat-history — retrieve previous AI conversations.
export function getChatHistory(): Promise<ChatHistoryEntry[]> {
  return apiRequest<ChatHistoryEntry[]>("/chat-history");
}

// GET /reports — retrieve previously generated reports.
export function getReports(): Promise<ReportSummary[]> {
  return apiRequest<ReportSummary[]>("/reports");
}
