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
> comparison logic is duplicated. Live data flows once /compare is exposed by the
> backend (app.py currently exposes only /predict) — same pattern as prior phases.

- [x] Comparison

## Phase 8 — AI Analysis
- [ ] Price Prediction
- [ ] Rental Analysis
- [ ] Property Valuation
- [ ] Risk Analysis
- [ ] Future Growth
- [ ] Investment Advisor
- [ ] Negotiation Strategy

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