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
> Demo backend endpoints (POST /login, POST /signup, GET /profile, POST /logout)
> are exposed via src/api/auth_api.py as a thin FastAPI router. These are demo
> stubs (no JWT, database, or password hashing) pending a future auth phase.

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
> Compare runs the documented POST /analysis/comparison endpoint and renders
> Property Score Cards + AI Recommendation + Comparison Table (UI doc §9). The
> comparison rendering (ComparisonResult) and tray logic are reused from Phases
> 4–6 — no comparison logic is duplicated in the frontend.
>
> Backend status: the EstateMind Copilot API (`src/api`) now exposes the
> `/analysis/comparison` endpoint as a thin FastAPI wrapper around the existing
> backend comparison service. All comparison logic continues to live in the
> existing backend (`property_tools.py`, `mcp_real_estate_service.py`,
> comparison agents, and ML services). The ML Prediction API (`app.py`)
> continues to run on port **8000** and is used internally by the EstateMind
> Copilot API when required. No business logic was duplicated or moved to the
> frontend.

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
- [x] Sign Out Button — POST /logout

## Phase 12 — Final Polish

> Error States (12.3): every API-consuming page now surfaces failures through the
> shared reusable ErrorState component (components/ui/error-state.tsx) — a clean,
> non-technical message plus a Retry button wired to the existing TanStack Query
> refetch (queries) or a re-run of the last request (mutations). Applied to AI
> Chat, Property Details, Property Comparison, AI Analysis, Reports (generate +
> share) and Saved Properties, and to the Profile sections via ProfileSection.
> Duplicated inline error banners were replaced with the shared component; the
> auth forms keep their existing form-level FormError (retry = resubmit the form).

> Empty States (12.4): every API-driven section now renders "nothing to show yet"
> through the shared reusable EmptyState component (components/ui/empty-state.tsx)
> — a friendly message, an appropriate icon and an optional CTA. Applied to AI
> Chat search results, the Evaluation Tray, Property Comparison, AI Analysis
> (placeholder + no-rows result), Reports, Saved Properties, the Profile sections
> (reports + chat history via ProfileSection) and the Property Details not-found
> state. Ad-hoc inline centered/dashed empty blocks were replaced with the shared
> component. Loading and error states were left unchanged; the chat composer keeps
> its richer SuggestedPrompts empty state (interactive prompt CTAs).

- [x] Responsive Design
- [x] Loading States
- [x] Error States
- [x] Empty States
- [x] UI Polish


# Phase 13 — Backend Integration

> Expose the existing EstateMind backend through FastAPI so the Next.js
> frontend communicates with the backend instead of the Streamlit UI.
> Reuse the existing services, MCP tools, agents, and ML models.
> No business logic should be duplicated or moved from the backend.

---

## Phase 13.1 — Chat API

> Expose the existing chat workflow by wrapping
> `parse_intent_and_execute()` inside a FastAPI endpoint.
>
> Refactor the chat service to remove direct Streamlit dependencies while
> remaining compatible with both the Streamlit UI and the FastAPI backend.

- [x] POST /chat 
- [x] Refactor `parse_intent_and_execute()` for Streamlit/FastAPI compatibility 

---

## Phase 13.2 — Property Details API

> Expose the existing property details service used by the Streamlit app.
> Reuse the existing `get_property_details()` backend function.

- [x] GET /property/{id} 

---

## Phase 13.3 — Report APIs

> Expose the existing report generation and report sharing workflow.
> Reuse the existing MCP tool (`send_property_report`) without changing
> the backend logic.
>
> Reports are generated as Markdown, previewed in the frontend,
> downloadable as Markdown, and can be shared through the existing
> n8n/WhatsApp workflow.

- [x] POST /report 
- [x] POST /report/share 
- [x] Create `create_property_report()` backend wrapper 

---

## Phase 13.4 — Authentication APIs

> Create authentication endpoints for the frontend.
> These are new APIs because the Streamlit application does not implement
> authentication.

- [x] POST /signup 
- [x] POST /login 
- [x] GET /profile 
- [x] POST /logout 

---

## Phase 13.5 — Saved Properties APIs

> Create endpoints for saving and retrieving bookmarked properties.

- [x] GET /saved-properties
- [x] POST /save-property
- [x] DELETE /save-property

---

## Phase 13.6 — Profile APIs

> Create endpoints for the user profile page.

- [x] GET /chat-history
- [x] GET /reports

---

## Phase 13.7 — FastAPI Integration

> Register all API routers in the FastAPI application and verify that
> every frontend endpoint is reachable through the backend.

- [x] Register all routers in `main.py`
- [x] Verify FastAPI application startup
- [x] Verify endpoint contracts with the frontend

---

# Phase 14 — Rich Property Experience

> Enhance the property browsing experience by reusing the existing backend
> property data. No backend business logic is modified.
>
> The backend already provides property metadata including images
> (`image_urls`) and the original property listing URL (`ap_pjt_url`).
>
> The frontend renders this information using reusable UI components.

---

## Phase 14.1 — Rich Property Cards

> Upgrade the reusable PropertyCard component using existing backend data.
>
> The single reusable PropertyCard (components/property/property-card.tsx) now
> renders rich property fields: primary image (image_urls[0]) with a fallback
> placeholder, BHK badge, bookmark, project name, property ID (links to
> /property/{id}), locality/city, bed/bath/parking/balcony, area, cost per sqft,
> formatted price, the backend recommendation (hybrid) score, and a Read More
> button. Every rich field is optional and rendered only when the backend
> provides it. The card is currently fed by the existing POST /chat
> `search_results` response, which returns only the core fields (id, price,
> bhk_type, location, amenities_mcp, search_score, why_recommended); the richer
> fields are ready for the documented POST /search contract
> (project_docs/03_API.md) once it is exposed, same pattern as prior phases. The
> bookmark reuses the existing saved-property workflow and no backend logic was
> changed or duplicated.

- [x] Display Property ID
- [x] Display BHK badge
- [x] Display bookmark button
- [x] Display primary property image
- [x] Display locality
- [x] Display bed
- [x] Display bath
- [x] Display parking
- [x] Display balcony
- [x] Display area
- [x] Display cost per sqft
- [x] Display formatted price
- [x] Display Recommendation Score
- [x] Display Read More button

---

## Phase 14.2 — Property Image Gallery

> Display backend property images using the existing `image_urls` field.
>
> Built a single reusable PropertyImageGallery component
> (components/property/property-image-gallery.tsx) that renders only the backend
> `image_urls`. It provides the hero image (image_urls[0]), a horizontally
> scrollable thumbnail strip with a green active border, an image counter
> (current / total), previous/next navigation with wrap-around, a fullscreen
> viewer built on the existing radix Dialog primitive (same one used by the
> shared Sheet), keyboard navigation (← / → to navigate, Esc to close — active
> only while the viewer is open), mobile swipe gestures, lazy loading, a loading
> skeleton (reusing the shared Skeleton), and a graceful ImageOff fallback for
> missing/broken images. Edge cases (0, 1, many, and large collections) are
> handled without layout shift. This is the single reusable image component to be
> consumed by Property Details (Phase 14.4), the interactive map (Phase 14.6) and
> any future property image display — no page was redesigned in this phase.

- [x] Display image gallery
- [x] Display thumbnails
- [x] Display image counter
- [x] Support lazy loading
- [x] Display image placeholder
- [x] Display thumbnail strip
- [x] Support image carousel
- [x] Support fullscreen viewer
- [x] Support previous / next navigation
- [x] Support keyboard navigation
- [x] Support mobile swipe

---

## Phase 14.3 — Original Property Listing

> Allow users to open the original property listing using the existing
> `ap_pjt_url` field.
>
> A single reusable OriginalListingButton
> (components/property/original-listing-button.tsx) opens the backend
> `ap_pjt_url` in a new tab (target="_blank", rel="noopener noreferrer") with an
> external-link icon, reusing the existing Button component and design system. It
> performs lightweight client-side validation (well-formed absolute http(s) URL)
> and renders nothing when the URL is null/undefined/empty/whitespace/invalid, so
> no disabled button, placeholder text or broken link is ever shown. It is
> consumed as a small "View Listing" action in the reusable PropertyCard and as a
> prominent "View Original Listing" button in the Property Details header. The
> `ap_pjt_url` field was added as an optional property to the existing
> PropertyCardData and PropertyDetail types. No backend logic or API contract was
> changed; this is strictly a UI enhancement.

- [x] View Original Listing button
- [x] Open in new tab
- [x] Hide button when URL unavailable

---

## Phase 14.4 — Rich Property Details Experience

> Build a rich property details page using the existing backend property data.
> Render every field returned by `GET /property/{id}`.
> Automatically organize backend fields into logical UI sections.
> Reuse existing backend fields and UI components without introducing new business logic.
>
> The Property Details view (features/property/property-details.tsx) now renders
> the full backend record dynamically — it never assumes a fixed schema. A pure
> categorization util (features/property/property-fields.ts) buckets every
> backend key into the documented logical sections (Pricing, Property Overview,
> Property Specifications, Project Information, Nearby Places, Reviews, Ratings,
> Location, AI Insights) using keyword rules; any field that matches no rule is
> rendered under "Additional Information", so no backend field is ever dropped
> and the page adapts to future fields without code changes. The page layout is:
> hero gallery → header → quick highlights → pricing → overview → specifications
> → project → amenities → features → nearby → reviews → ratings → location →
> AI insights → additional information → view original listing.
>
> Reused components: PropertyImageGallery (14.2) for hero/thumbnail/fullscreen/
> remaining images, OriginalListingButton (14.3) in the header and as a
> prominent action, the shared UI primitives, the format helpers, and the
> saved-property workflow (header bookmark). A reusable PropertyLocationMap
> placeholder (components/property/property-location-map.tsx) consumes the
> backend latitude/longitude and exposes external map links — the full
> interactive property map remains Phase 14.6 and is NOT implemented here.
> `PropertyDetail` is now an open record (typed core fields + index signature),
> so rich fields render only when the backend provides them. No backend logic or
> API contract was changed; this is strictly a frontend rendering improvement.

- [x] Hero Image
- [x] Image Gallery
- [x] Thumbnail Carousel
- [x] Fullscreen Viewer
- [x] Property Header
- [x] Quick Highlights
- [x] Property Overview
- [x] Pricing Section
- [x] Property Specifications
- [x] Project Information
- [x] Amenities Grid
- [x] Features Section
- [x] Nearby Places
- [x] Positive Reviews
- [x] Needs Improvement
- [x] Ratings
- [x] Project Ratings
- [x] Review Cards
- [x] Location Information
- [x] Additional Information
- [x] Original Listing Button
- [x] AI Insights
- [x] Interactive Map — reusable location placeholder consuming lat/lng (full map is Phase 14.6)
- [x] Responsive Layout
- [x] Render all backend property fields
- [x] Automatically organize unknown backend fields

---

## Phase 14.5 — End-to-End Testing

> End-to-end verification of every feature implemented through Phase 14.4. This
> phase was testing/bug-fixing only — no features, endpoints or workflows were
> added, and no backend business logic or API contracts were changed.
>
> Verified: TypeScript compiles with no errors (`tsc --noEmit`), ESLint passes
> with no warnings/errors, and the production build (`next build`) succeeds and
> generates every route (/, /login, /signup, /dashboard, /chat, /compare,
> /analysis, /reports, /saved, /profile, /property/[id]). Reviewed the reusable
> Phase 14 components for correctness and design-system consistency:
> PropertyCard (rich fields render only when the backend provides them; bookmark
> + Read More navigation to /property/{id}), PropertyImageGallery (hero,
> thumbnails, counter, prev/next wrap-around, fullscreen viewer with Esc + arrow
> keys, mobile swipe, lazy loading, skeleton, ImageOff fallback, no layout
> shift), OriginalListingButton (opens ap_pjt_url in a new tab with
> target="_blank" rel="noopener noreferrer"; hidden for missing/invalid URLs),
> the dynamic PropertyDetails page (renders every backend field, categorizes
> unknown fields into Additional Information), and PropertyLocationMap. No
> console statements remain in the frontend source; loading/success/empty/error
> states reuse the shared EmptyState/ErrorState components. No confirmed bugs
> were found; no code changes were required.

- [x] Property Card rendering
- [x] Image gallery
- [x] Fullscreen viewer
- [x] Image navigation
- [x] Original listing
- [x] Mobile responsiveness

---

## Phase 14.6 — Interactive Property Map

> Visualize property locations using backend latitude and longitude.
> Reuse backend property data without introducing new business logic.
>
> Built one reusable InteractivePropertyMap component
> (components/property/interactive-property-map.tsx) — the single map used across
> the application. It consumes only backend data (latitude/longitude + optional
> rich fields) and never invents coordinates or renders mock markers. Leaflet is
> client-only, so the react-leaflet implementation
> (components/property/property-map-view.tsx) is code-split with next/dynamic
> (ssr:false) behind a Skeleton; the public wrapper filters out
> missing/invalid/(0,0) coordinates and hides the map gracefully (renders
> nothing) when there is nothing valid to show, so no broken container appears.
>
> Markers show only the formatted price (compact ₹ Cr / Lakh via a new
> formatPriceLabel helper in features/dashboard/format.ts). Selecting a marker
> opens a rich preview card (primary image from image_urls, project name,
> locality/city, formatted price, area, cost per sqft, BHK) with a Read More
> button that reuses the same /property/[id] navigation as the Property Cards.
> The map auto-fits its bounds to every marker and re-fits when the property list
> changes; a single property centres at a comfortable zoom. Marker clustering
> (leaflet.markercluster with chunkedLoading) keeps rendering efficient for large
> datasets (~11k). Zoom, pan, marker selection/deselection and responsive
> resizing (ResizeObserver → invalidateSize) are supported.
>
> Reused on Search Results (features/dashboard/search-results-panel.tsx) above
> the reusable PropertyCard grid with marker↔card highlight sync, and on Property
> Details (features/property/property-details.tsx) for the single selected
> property, with the existing PropertyLocationMap placeholder as the
> no-coordinates fallback. The full PropertyImageGallery (14.2) remains the image
> component reached via Read More; the popup shows the same primary-image
> rendering as the cards. SearchResult was extended with the optional map/rich
> fields from the documented POST /search contract (same pattern as prior
> phases), so real markers appear once /search is exposed and the map hides until
> then. react-leaflet-cluster was avoided (React-18 peer); clustering uses the
> framework-agnostic leaflet.markercluster via a thin createPathComponent binding
> that is React-19 compatible. No backend logic or API contract was changed; no
> duplicate map component was created.

- [x] Display interactive property map
- [x] Display all properties using price-only markers
- [x] Display filtered search results on the map
- [x] Display rich property markers
- [x] Show property image
- [x] Show formatted price
- [x] Show cost per sqft
- [x] Show area
- [x] Show BHK
- [x] Navigate to Property Details
- [x] Reuse Image Gallery from Property Cards
- [x] Support zoom and pan
- [x] Support marker clustering for large datasets
- [x] Mobile responsive
- [x] Enable marker clustering for large property datasets (~11k properties)

---

## Phase 14.7 — Landing Page Redesign

> Redesign the public landing page with a premium first impression.
> Use a large scenic real estate background image similar to the provided design reference.
> Preserve existing authentication workflows while modernizing the visual experience.
>
> The public Landing Page (app/page.tsx) was redesigned into a premium,
> AI-first real estate landing experience. The Hero
> (features/landing/hero.tsx) is a full-screen section with a scenic
> real-estate background image (loaded via CSS so no per-domain Next image
> config is needed, matching the app's existing plain-<img> external-image
> pattern) under a dark gradient overlay, EstateMind Copilot branding, a
> welcome heading, an AI-powered subtitle, a short description, three primary
> actions (Sign Up → /signup, Sign In → /login, Learn More → #features) and a
> bouncing scroll-to-features indicator. The Features section
> (features/landing/features.tsx) introduces existing backend capabilities
> only (AI Property Search, Property Comparison, Price Prediction, Rental
> Analysis, Property Valuation, Risk Analysis, Future Growth Analysis,
> Investment Advisor, AI Report Generation) through a single reusable
> FeatureCard (features/landing/feature-card.tsx). "How EstateMind works"
> (features/landing/how-it-works.tsx) renders the Search → Compare → Analyze
> → Generate Report → Make Better Decisions journey via a single reusable
> WorkflowStep (features/landing/workflow-step.tsx). The final CallToAction
> (features/landing/call-to-action.tsx) offers Sign In + Sign Up with a subtle
> hover animation.
>
> Only the Landing Page was touched: authentication logic, backend APIs and
> authenticated pages are unchanged. All buttons reuse the existing Button
> component and the app's theme tokens (which become green under Phase 14.8's
> theme refresh), so the page stays consistent with one design system. No
> duplicate components were created; the LandingNavbar and LandingFooter were
> reused as-is. TypeScript compiles, ESLint passes, and `next build` succeeds
> with `/` prerendered as static.

- [x] Fullscreen Hero Background
- [x] Background Hero Image
- [x] Dark Overlay
- [x] EstateMind Branding
- [x] Welcome Section
- [x] Sign In Button
- [x] Sign Up Button
- [x] Animated Call To Action
- [x] AI Features Section
- [x] Feature Cards
- [x] How EstateMind Works
- [x] Premium Landing Page
- [x] Responsive Design

---

## Phase 14.8 — UI Theme Refresh

> Refresh the application's visual design using the provided green-and-white dashboard reference.
> Update only the design system and styling. Do not modify functionality or existing workflows.

- [ ] Green Primary Palette
- [ ] Update all colors to green palette
- [ ] White Background
- [ ] Light Gray Cards
- [ ] Premium Cards
- [ ] Update Sidebar
- [ ] Update Navbar
- [ ] Update Property Cards
- [ ] Update Cards
- [ ] Update Badges
- [ ] Update Forms
- [ ] Update Buttons
- [ ] Update Tables
- [ ] Softer Borders
- [ ] Better Typography
- [ ] Better Spacing
- [ ] Rounded Corners
- [ ] Soft Shadows
- [ ] Green Active Navigation
- [ ] Green Highlights for Important Actions
- [ ] Keep one consistent design system
- [ ] Responsive Polish

---

# Phase 15 — AI Conversational Copilot

> Extend the existing EstateMind Copilot by integrating Claude as a
> natural-language reasoning layer on top of the existing backend.
>
> Claude does **not** replace the Machine Learning models, recommendation
> engine, or analysis agents.
>
> The existing backend remains the single source of truth for:
>
> - Property Search
> - Recommendation Engine
> - Price Prediction
> - Risk Analysis
> - Rental Analysis
> - Future Growth Analysis
> - Negotiation Strategy
> - Property Comparison
> - Report Generation
>
> Claude only explains, summarizes and answers user questions using the
> structured output already produced by the backend.
>
> No business logic should be duplicated inside Claude.

---

## Phase 15.1 — Claude API Integration

> Integrate the Anthropic Claude API into the EstateMind backend.
> Create a reusable Claude service that can be called by existing FastAPI
> endpoints without changing backend business logic.

- [ ] Configure Claude API
- [ ] Create Claude service
- [ ] Add environment configuration
- [ ] Verify Claude connectivity

---

## Phase 15.2 — Prompt Builder

> Create reusable prompt builders that transform backend analysis into
> structured prompts for Claude.

- [ ] Build Search Prompt
- [ ] Build Property Analysis Prompt
- [ ] Build Comparison Prompt
- [ ] Build Report Prompt

---

## Phase 15.3 — AI Search Explanation

> After the existing recommendation engine returns search results,
> Claude generates a conversational explanation describing why the
> properties were recommended.

Flow

User

↓

Search Agent

↓

Recommendation Engine

↓

Claude

↓

Natural-language explanation

↓

Frontend

- [ ] Explain search results
- [ ] Summarize recommended properties
- [ ] Explain ranking decisions

---

## Phase 15.4 — AI Property Analysis

> Combine the existing backend analysis agents and let Claude explain the
> results in natural language.

Existing backend agents remain unchanged.

- [ ] Explain Risk Analysis
- [ ] Explain Rental Analysis
- [ ] Explain Future Growth
- [ ] Explain Valuation
- [ ] Explain Negotiation Strategy

---

## Phase 15.5 — AI Property Comparison

> Use the existing comparison agent to determine the best property and let
> Claude explain the strengths and weaknesses of each option.

- [ ] Explain comparison results
- [ ] Summarize investment advantages
- [ ] Generate recommendation summary

---

## Phase 15.6 — AI Investment Advisor

> Combine all existing backend analysis into a single structured context
> and let Claude generate a complete investment recommendation.

Existing backend performs:

- Search
- Recommendation
- Prediction
- Rental Analysis
- Risk Analysis
- Future Growth
- Negotiation
- Valuation

Claude only explains the combined result.

- [ ] Investment summary
- [ ] Pros and Cons
- [ ] Final recommendation
- [ ] Investment reasoning

---

## Phase 15.7 — Conversational Memory

> Maintain conversation context during a user session.

Claude should remember:

- Previous searches
- Evaluation tray
- Compared properties
- User preferences
- Previous follow-up questions

- [ ] Session memory
- [ ] Follow-up questions
- [ ] Multi-turn conversation

---

## Phase 15.8 — Intelligent Tool Orchestration

> Allow Claude to decide which existing backend capability should be
> executed based on the user's request.

Claude may invoke:

- Search
- Comparison
- Prediction
- Rental Analysis
- Risk Analysis
- Future Growth
- Negotiation
- Report Generation
- Report Sharing
- Saved Properties

No backend business logic is moved into Claude.

- [ ] Search tool
- [ ] Comparison tool
- [ ] Analysis tools
- [ ] Report tools
- [ ] Saved-property tools

---

## Phase 15.9 — Streaming Responses

> Stream Claude responses to the frontend for a more responsive chat
> experience.

- [ ] Streaming API
- [ ] Streaming frontend support
- [ ] Loading indicators

---

## Phase 15.10 — AI Report Enhancement

> Enhance the existing backend-generated reports by allowing Claude to
> produce professional summaries.

Backend generates the report.

Claude improves readability.

- [ ] Executive summary
- [ ] Investment summary
- [ ] Risk summary
- [ ] Recommendation summary

---

## Phase 15.11 — AI Suggestions

> Claude suggests useful follow-up actions after every conversation.

Examples:

- Compare these properties
- Generate report
- Analyze rental income
- View property details
- Save property

- [ ] Suggested actions
- [ ] Follow-up recommendations

---

## Phase 15.12 — End-to-End Testing

> Verify the complete conversational workflow.

- [ ] Search → Claude explanation
- [ ] Property Details → Claude explanation
- [ ] Comparison → Claude explanation
- [ ] Investment Advisor → Claude explanation
- [ ] Report Generation → Claude summary
- [ ] Report Sharing
- [ ] Session memory
- [ ] Streaming responses
- [ ] Error handling

---

## Future Enhancements

> These enhancements are outside the current project scope and can be
> implemented after the core application is complete.

- [ ] Export reports as PDF
- [ ] Export reports as DOCX
- [ ] Authentication with JWT
- [ ] Persistent database for saved properties
- [ ] Persistent chat/session storage
- [ ] Multi-model AI support (Claude, GPT, Gemini)
- [ ] Voice-based property assistant
- [ ] Image-based property analysis