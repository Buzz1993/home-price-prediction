// Profile page types. `GET /profile` returns the same account shape as the shared
// auth `User`, so it is re-exported here. The chat-history and report-list shapes
// mirror the documented GET /chat-history and GET /reports contracts
// (project_docs/03_API.md). They are kept lenient because the EstateMind Copilot
// API (src/api) does not yet expose these endpoints — the frontend targets the
// documented contract, matching the Phase 4–10 approach.

import type { User } from "@/types/auth";

export type { User };

// One previous AI conversation turn returned by GET /chat-history. Mirrors the
// backend chat message concept (role + text).
export type ChatHistoryEntry = {
  id?: string;
  role?: "user" | "assistant";
  content: string;
};

// One previously generated report returned by GET /reports. `property_ids` are
// the properties the report covers; `summary` is an optional short preview.
export type ReportSummary = {
  id: string;
  title?: string;
  property_ids?: string[];
  summary?: string;
  created_at?: string;
};
