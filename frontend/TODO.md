# EstateMind Frontend TODO

## Phase 1 — Project Setup
- [x] Initialize Next.js
- [x] Configure TypeScript
- [x] Configure Tailwind CSS
- [x] Install shadcn/ui
- [x] Configure TanStack Query
- [x] Configure ESLint
- [x] Create folder structure
- [x] Configure environment variables

## Phase 2 — Layout
- [x] Navbar
- [x] Sidebar
- [x] Dashboard Layout
- [x] Responsive Navigation

## Phase 3 — Authentication

> Authentication UI is complete.
> Backend authentication endpoints (/login, /signup, /profile) are planned but not yet implemented.
> Frontend integration will be completed once those endpoints are available.

- [x] Landing Page
- [x] Login
- [x] Signup

## Phase 4 — Dashboard

> Dashboard workspace UI is complete.
> It consumes the documented POST /chat endpoint, which is not yet exposed by
> the FastAPI backend (app.py currently exposes only /predict). Live data will
> flow once /chat is available — same pattern as the Phase 3 auth pages.

- [x] Dashboard Page
- [x] AI Chat Workspace
- [x] Search Results Panel
- [x] Evaluation Tray

## Phase 5 — AI Chat

> Dedicated /chat route reuses the Phase 4 Copilot workspace (chat + evaluation
> tray) via the shared CopilotWorkspace component — no chat logic duplicated.
> Search results now render as reusable PropertyCards instead of a raw table.
> Live data flows once the documented POST /chat endpoint is exposed by the
> backend (app.py currently exposes only /predict) — same pattern as Phase 4.

- [x] AI Chat
- [x] Search Results
- [x] Property Cards

## Phase 6 — Property Details

> Property Details page (/property/[id]) reuses the shared UI primitives and the
> format/splitList helpers to render the full property record — gallery
> placeholder, information, amenities, features, location and price. Property
> cards now link to it. It consumes the documented GET /property/{id} endpoint,
> which is not yet exposed by the backend (app.py currently exposes only
> /predict) — live data flows once it is available, same pattern as prior
> phases. AI analysis sections (price/risk/rental/valuation) belong to Phase 8.

- [x] Property Details

## Phase 7 — Property Comparison

> Dedicated Property Comparison page (/compare) reuses the shared evaluation tray
> — WorkspaceProvider is lifted to the (dashboard) layout so a tray staged in AI
> Chat persists onto this page. Selecting ≥2 staged properties and pressing
> Compare runs the documented POST /compare endpoint and renders Property Score
> Cards + AI Recommendation + Comparison Table (UI doc §9). The comparison
> rendering (ComparisonResult) and tray logic are reused from Phases 4–6 — no
> comparison logic is duplicated. Live data flows once /analysis/comparison is exposed by the
> backend (app.py currently exposes only /predict) — same pattern as prior phases.

- [x] Property Comparison — POST /analysis/comparison

## Phase 8 — AI Analysis

> Dedicated AI Analysis page (/analysis) reuses the shared evaluation tray to
> pick staged properties, then runs each analysis and renders the result with
> the existing reusable renderers (AnalysisTable / AdvisorCards /
> NegotiationCards) — no analysis logic is duplicated in the frontend.
>
> Backend status: The EstateMind Copilot API (`src/api`) now exposes the
> `/analysis/*` endpoints as thin FastAPI wrappers around the existing backend
> services. All analysis logic continues to live in the existing backend
> (`property_tools.py`, `mcp_real_estate_service.py`, analysis agents, and ML
> services). The ML Prediction API (`app.py`) continues to run on port **8000**
> and is used internally by the EstateMind Copilot API when required. No
> business logic was duplicated or moved to the frontend.

- [x] Price Prediction — POST /analysis/predict
- [x] Rental Analysis — POST /analysis/rental
- [x] Property Valuation — POST /analysis/valuation
- [x] Risk Analysis — surfaced from POST /analysis/advisor
- [ ] Future Growth — BLOCKED: no backend endpoint or MCP tool is currently available.
- [x] Investment Advisor — POST /analysis/advisor
- [x] Negotiation Strategy — POST /analysis/negotiation

## Phase 9 — Reports

> Dedicated Reports page (/reports) reuses the shared evaluation tray to pick the
> staged properties a report covers, then generates an AI report (POST /report),
> previews it, downloads it as Markdown (client-side Blob), and shares it to a
> phone number (POST /report/share) — reproducing the Streamlit report workflow
> (select → generate → preview → share). Tray + UI primitives are reused; no
> report logic is duplicated in the frontend (the backend composes the report and
> delivers it via the MCP tool send_property_report + n8n).
>
> Backend status: the EstateMind Copilot API (src/api) currently exposes only the
> /analysis/* endpoints — neither /report nor /report/share is wired yet (the
> report-share logic already exists as the MCP tool send_property_report in
> src/mcp/tools/property_tools.py but is not exposed). Same pattern as Phases 4–8:
> each action is wired to its documented POST endpoint and live data flows once
> the backend exposes it. No backend was modified and no endpoint was invented.

- [x] Report Generation — POST /report
- [x] Report Sharing — POST /report/share

## Phase 10 — Saved Properties

> Dedicated Saved Properties page (/saved) lists the user's saved properties via
> GET /saved-properties and renders each as a reusable PropertyCard. The card now
> carries an optional bookmark toggle: saving from AI Chat search results
> (POST /save-property) and removing from the saved page (DELETE /save-property).
> Saved state is shared through TanStack Query — mutations invalidate the
> saved-properties query so search results and the saved list stay in sync.
> Staging is reused from the shared evaluation tray, so a saved property can flow
> into comparison, analysis or a report. No persistence logic is duplicated in the
> frontend — the backend owns the saved list.
>
> Backend status: the EstateMind Copilot API (src/api) currently exposes only the
> /analysis/* endpoints — none of the saved-properties endpoints are wired yet
> (there is also no Streamlit reference for this feature). Same pattern as Phases
> 4–9: each action targets its documented contract (project_docs/03_API.md) and
> live data flows once the backend exposes it. No backend was modified and no
> endpoint was invented.

- [x] Saved Properties — GET /saved-properties, POST /save-property, DELETE /save-property

## Phase 11 — Profile

> Dedicated Profile page (/profile) shows the user's account, generated reports and
> AI chat history, plus a logout action (UI doc §15). Each section reads its own
> documented endpoint (GET /profile, GET /reports, GET /chat-history) and handles
> loading, empty and error states through the reusable ProfileSection wrapper. The
> chat history reuses the shared ChatMessage bubble; the account section reuses the
> auth session for logout (clears the client-side session via auth-provider, drops
> cached queries, and redirects to /login) — no authentication or business logic is
> implemented in the frontend.
>
> Backend status: the EstateMind Copilot API (src/api) currently exposes only the
> /analysis/* endpoints — none of the profile endpoints are wired yet (there is also
> no Streamlit reference for this feature). Same pattern as Phases 4–10: each query
> targets its documented contract (project_docs/03_API.md) and live data flows once
> the backend exposes it. No backend was modified and no endpoint was invented.

- [x] User Profile — GET /profile, GET /chat-history, GET /reports

## Phase 12 — Final Polish

> Error States (12.3): every API-consuming page now surfaces failures through the
> shared reusable ErrorState component (components/ui/error-state.tsx) — a clean,
> non-technical message plus a Retry button wired to the existing TanStack Query
> refetch (queries) or a re-run of the last request (mutations). Applied to AI
> Chat, Property Details, Property Comparison, AI Analysis, Reports (generate +
> share) and Saved Properties, and to the Profile sections via ProfileSection.
> Duplicated inline error banners were replaced with the shared component; the
> auth forms keep their existing form-level FormError (retry = resubmit the form).

- [x] Responsive Design
- [x] Loading States
- [x] Error States
- [ ] Empty States
- [ ] UI Polish