// Report History model + persistence (Phase 18.9).
//
// Whenever a report is generated successfully, its metadata AND the already
// generated report text are stored locally so previously generated reports can
// be reopened, downloaded, shared and deleted without regenerating them.
// This mirrors the existing conversation persistence (conversations.ts):
// client-only localStorage, best-effort, bounded — no backend, API contract or
// database is involved, and report generation itself is untouched.

export type ReportType = "property" | "comparison";

export type StoredReport = {
  id: string;
  // "Property Report" | "Comparison Report" — derived from the type.
  title: string;
  type: ReportType;
  // Epoch ms when the report was generated (drives date/time display).
  createdAt: number;
  // The properties the report covers.
  propertyIds: string[];
  // The EXACT report text the user previewed (Claude-enhanced when available,
  // otherwise the unchanged backend report). Stored so Preview / Download /
  // Share reuse it verbatim — reports are never regenerated.
  content: string;
  // Whether `content` is the Claude-enhanced presentation (status badge).
  aiEnhanced: boolean;
};

const STORAGE_KEY = "estatemind:reports:v1";
// Keep the persisted history bounded (each entry carries full report text).
const MAX_REPORTS = 30;

export function reportTitle(type: ReportType): string {
  return type === "comparison" ? "Comparison Report" : "Property Report";
}

// Load the persisted report history (client only), newest first. Returns []
// when nothing is stored or the payload is unreadable.
export function loadReportHistory(): StoredReport[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as StoredReport[];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((r) => r && typeof r.id === "string" && typeof r.content === "string")
      .sort((a, b) => b.createdAt - a.createdAt);
  } catch {
    return [];
  }
}

function persist(reports: StoredReport[]): void {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(reports.slice(0, MAX_REPORTS))
    );
  } catch {
    // Ignore quota / serialization errors — persistence is best-effort.
  }
}

// Store a newly generated report at the top of the history and return the
// updated list (newest first).
export function addReport(input: {
  type: ReportType;
  propertyIds: string[];
  content: string;
  aiEnhanced: boolean;
}): StoredReport[] {
  const entry: StoredReport = {
    id:
      globalThis.crypto?.randomUUID?.() ??
      `report-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    title: reportTitle(input.type),
    type: input.type,
    createdAt: Date.now(),
    propertyIds: input.propertyIds,
    content: input.content,
    aiEnhanced: input.aiEnhanced,
  };
  const next = [entry, ...loadReportHistory()].slice(0, MAX_REPORTS);
  persist(next);
  return next;
}

// Remove a report by id and return the updated list.
export function deleteReport(id: string): StoredReport[] {
  const next = loadReportHistory().filter((r) => r.id !== id);
  persist(next);
  return next;
}
