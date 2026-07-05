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
