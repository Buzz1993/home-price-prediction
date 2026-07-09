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
  // Original property listing URL. Optional — rendered only when present.
  ap_pjt_url?: string;
};

// Minimal shape the reusable PropertyCard needs to render one property. A full
// SearchResult satisfies it directly; saved properties are mapped onto it from
// GET /property/{id}, where `search_score`/`why_recommended` are unavailable.
//
// The card is currently fed by the existing POST /chat `search_results`
// response, which returns only the core fields (id, price, bhk_type, location,
// amenities_mcp, search_score, why_recommended). The optional rich fields below
// are ready for the documented POST /search contract (project_docs/03_API.md),
// which is not yet exposed, so the card renders each rich field only when the
// backend provides it and hides it otherwise. `search_score` carries the
// backend hybrid (recommendation) score.
export type PropertyCardData = {
  id: string;
  price: number;
  bhk_type: string;
  location: string;
  amenities_mcp: string;
  why_recommended?: string;
  search_score?: number;
  image_urls?: string[];
  project_name?: string;
  locality?: string;
  city?: string;
  bed?: number | string;
  bath?: number | string;
  parking?: number | string;
  balcony?: number | string;
  area?: number | string;
  cost_per_sqft?: number | string;
  // Original property listing URL. Optional — rendered only when present.
  ap_pjt_url?: string;
};

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

// Request body for POST /chat (backend ChatRequest in src/api/chat_api.py). The
// staged tray travels with the message so the backend can run tray-based
// analyses (compare, rental, prediction, …). `slider_weights` is optional and
// unused by the frontend, so it is omitted.
export type ChatRequest = {
  message: string;
  staged_property_ids: string[];
};

// A rendered conversation entry. `text` is the header/message; when the entry
// carries structured data, `response` holds the typed payload to render.
export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
};
