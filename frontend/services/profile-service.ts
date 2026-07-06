// Profile API calls. Thin wrappers over the documented Profile endpoints
// (GET /profile, GET /chat-history, GET /reports — see project_docs/03_API.md).
// The backend owns the user account, conversation history and report list; the
// frontend only reads and renders them.
//
// Backend limitation: the EstateMind Copilot API (src/api) currently exposes only
// the /analysis/* endpoints — none of these profile endpoints are wired yet. These
// wrappers target the documented contract so live data flows once the endpoints
// are exposed, matching the Phase 4–10 pattern. No API is invented and no business
// logic lives in the frontend.

import { apiRequest } from "@/lib/api-client";
import type { ChatHistoryEntry, ReportSummary, User } from "@/types/profile";

// GET /profile — retrieve the current user's profile.
export function getProfile(): Promise<User> {
  return apiRequest<User>("/profile");
}

// GET /chat-history — retrieve previous AI conversations.
export function getChatHistory(): Promise<ChatHistoryEntry[]> {
  return apiRequest<ChatHistoryEntry[]>("/chat-history");
}

// GET /reports — retrieve previously generated reports.
export function getReports(): Promise<ReportSummary[]> {
  return apiRequest<ReportSummary[]>("/reports");
}
