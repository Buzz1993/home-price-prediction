// Dashboard workspace types. Mirror the shapes returned by the existing backend
// chat pipeline (src/services/chat_service.py -> parse_intent_and_execute and
// src/mcp/tools/property_tools.py). The frontend only renders these; all
// business logic stays in the backend.

// A single ranked property returned by a search (`type: "search_results"`).
export type SearchResult = {
  id: string;
  price: number;
  bhk_type: string;
  location: string;
  amenities_mcp: string;
  search_score: number;
  why_recommended: string;
};

// Full property record returned by GET /property/{id}. Mirrors the backend
// PROPERTY_DETAIL_WHITELIST (src/mcp/tools/property_tools.py -> get_property_details).
export type PropertyDetail = {
  id: string;
  project_name: string;
  builder: string;
  location: string;
  price: number;
  area: number | string;
  bhk_type: string;
  amenities_mcp: string;
  features_mcp: string;
  analysis_msg: string;
};

// A saved (favourited) property returned by GET /saved-properties. Modeled on
// SearchResult so the reusable PropertyCard renders it directly. Kept as an
// alias because the backend does not yet expose the saved-properties endpoints
// (see project_docs/03_API.md) — the display shape mirrors search results.
export type SavedProperty = SearchResult;

// The backend returns this shape (HTTP 200) when no property matches the id.
export type NotFoundResponse = { error: string };

// One row of a comparison ranking / the winning property.
export type ComparisonRow = {
  id: string;
  overall_score: number;
  verdict: string;
  comparison_reason: string;
};

export type ComparisonResult = {
  winner: ComparisonRow;
  rankings: ComparisonRow[];
};

// Rental, prediction and valuation results are rendered as tables. Their exact
// columns come from the backend, so keep them as open records.
export type AnalysisRow = Record<string, string | number | null>;

export type NegotiationRow = {
  id: string;
  target_price: number | string;
  suggested_discount_percent: number | string;
  negotiation_power: string;
  strategy: string;
  talking_points: string;
};

export type AdvisorRow = {
  id: string;
  suitable_for: string;
  verdict: string;
  positives: string;
  risks: string;
};

// Result returned by POST /report. The backend composes an AI property report
// (text / markdown) for the given properties; the frontend only previews,
// downloads and shares it. `report` is the text the backend also feeds to its
// n8n share tool (send_property_report). Kept lenient because the backend does
// not yet expose /report — see project_docs/03_API.md.
export type ReportResult = {
  report: string;
  property_ids?: string[];
};

// Delivery status returned by POST /report/share. Mirrors the backend
// send_property_report return shape ({ status, status_code }).
export type ShareResult = {
  status: string;
  status_code?: number;
};

// Discriminated union of every response the /chat endpoint can return. The
// `type` field matches RESPONSE_CONFIG keys in the Streamlit reference.
export type ChatResponse =
  | { type: "text"; content: string }
  | { type: "search_results"; content: SearchResult[] }
  | { type: "comparison"; content: ComparisonResult }
  | { type: "rental"; content: AnalysisRow[] }
  | { type: "prediction"; content: AnalysisRow[] }
  | { type: "negotiation"; content: NegotiationRow[] }
  | { type: "valuation"; content: AnalysisRow[] }
  | { type: "advisor"; content: AdvisorRow[] };

export type ChatResponseType = ChatResponse["type"];

// Request body for POST /chat. The tray travels with the message so the backend
// can run tray-based analyses (compare, rental, prediction, …).
export type ChatRequest = {
  message: string;
  tray: string[];
};

// A rendered conversation entry. `text` is the header/message; when the entry
// carries structured data, `response` holds the typed payload to render.
export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
};
