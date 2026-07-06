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
> Backend status (verified via the running FastAPI at http://localhost:8000):
> app.py exposes only /, /health and /predict. The documented per-property
> analysis endpoints (/rental, /valuation, /advisor, /negotiation) return 404,
> and /predict currently expects a full raw property record (the Data model),
> not a Property ID. The analysis logic itself already exists as backend MCP
> tools (get_price_prediction, get_rental_analysis, get_valuation_analysis,
> get_negotiation_strategy, get_investment_advice) but is only reachable through
> the Streamlit /chat routing. So — same pattern as Phases 4–7 — each analysis is
> wired to its documented POST endpoint and live data flows once the backend
> exposes it. No backend was modified and no endpoint was invented.

- [x] Price Prediction — POST /analysis/predict
- [x] Rental Analysis — POST /analysis/rental
- [x] Property Valuation — POST /analysis/valuation
- [x] Risk Analysis — surfaced from POST /analysis/advisor
- [ ] Future Growth — BLOCKED: no backend endpoint or MCP tool is currently available.
- [x] Investment Advisor — POST /analysis/advisor
- [x] Negotiation Strategy — POST /analysis/negotiation

## Phase 9 — Reports
- [ ] Report Generation
- [ ] Report Sharing

## Phase 10 — Saved Properties
- [ ] Saved Properties

## Phase 11 — Profile
- [ ] User Profile

## Phase 12 — Final Polish
- [ ] Responsive Design
- [ ] Loading States
- [ ] Error States
- [ ] Empty States
- [ ] UI Polish