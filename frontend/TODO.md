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
- [x] Future Growth — surfaced from POST /analysis/advisor (growth_label / growth_reason produced by run_future_agent during enrichment; analysis_type=future for the AI explanation)
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
>
> This phase was a shared-theme refresh only — no functionality, workflow, layout,
> component logic, API contract or backend was changed. The entire application is
> token-driven (shadcn primitives + shared layout/UI components read the CSS
> variables in `app/globals.css`), so the refresh was centralized there rather
> than styled page-by-page. `:root` now defines the premium green-and-white
> EstateMind system: white background, a professional green primary (≈ emerald
> 700, AA contrast on white), light-gray cards, soft gray borders, a green focus
> ring, faint-green accent/hover surfaces, black primary + gray secondary text, a
> green chart scale, green sidebar primary/active/ring tokens, and a slightly
> larger corner radius (0.7rem) for a softer, premium feel. The `.dark` block was
> kept internally consistent (green primary/ring/sidebar/charts) even though the
> app runs in light mode. A small base-layer typography pass adds heading
> tracking + `text-wrap: balance`, body `text-wrap: pretty`, and legibility
> smoothing.
>
> Because primary/accent/ring/sidebar tokens flow through every primitive, this
> automatically greens: primary buttons, link buttons, default badges, active
> sidebar navigation, the brand mark, focus rings on inputs/buttons, table row
> selection, and every card/panel — with one consistent radius, soft shadow and
> spacing scale inherited from the existing components. Semantic colors were left
> intact and now sit naturally within the green family: emerald = success/winner,
> amber = warning, destructive = red errors. No duplicate components or
> page-specific themes were created. TypeScript compiles, ESLint passes, and
> `next build` prerenders every route.

- [x] Green Primary Palette
- [x] Update all colors to green palette
- [x] White Background
- [x] Light Gray Cards
- [x] Premium Cards
- [x] Update Sidebar
- [x] Update Navbar
- [x] Update Property Cards
- [x] Update Cards
- [x] Update Badges
- [x] Update Forms
- [x] Update Buttons
- [x] Update Tables
- [x] Softer Borders
- [x] Better Typography
- [x] Better Spacing
- [x] Rounded Corners
- [x] Soft Shadows
- [x] Green Active Navigation
- [x] Green Highlights for Important Actions
- [x] Keep one consistent design system
- [x] Responsive Polish

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

- [x] Configure Claude API
- [x] Create Claude service
- [x] Add environment configuration
- [x] Verify Claude connectivity

---

## Phase 15.2 — Prompt Builder

> Create reusable prompt builders that transform backend analysis into
> structured prompts for Claude.

- [x] Build Search Prompt
- [x] Build Property Analysis Prompt
- [x] Build Comparison Prompt
- [x] Build Report Prompt

---

## Phase 15.3 — AI Search Explanation

> After the existing recommendation engine returns search results,
> Claude generates a conversational explanation describing why the
> properties were recommended.
>
> The existing backend still performs intent extraction, property search,
> hybrid ranking and recommendation exactly as before
> (chat_service.parse_intent_and_execute) — its search results are returned
> unchanged. A thin explanation service (src/llm/search_explanation.py) reuses
> the Phase 15.2 Search Prompt Builder (build_search_prompt) and the Phase 15.1
> Claude Client (ask_claude) to turn that structured result into a
> natural-language explanation of why those properties were recommended. No
> business logic, ranking or recommendation logic was duplicated or changed, and
> no new REST endpoint was added.
>
> The existing POST /chat controller (src/api/chat_api.py) attaches the
> explanation as an optional, additive `ai_explanation` field on search_results
> responses only; `content` (the ranked properties) is never modified. Claude is
> optional — if it fails or is unavailable, the field is simply omitted and the
> search results still return, so search never breaks.
>
> The frontend reuses the existing AI Chat interface: a single reusable
> SearchExplanation card (features/dashboard/search-explanation.tsx) renders the
> explanation above the existing Property Cards inside the chat message, and
> shows a graceful "AI explanation is temporarily unavailable" message when the
> backend omits it. The Property Cards, Evaluation Tray, search workflow and
> existing loading state are unchanged; the search_results response type gained
> only the optional `ai_explanation` field. tsc, ESLint and next build all pass.

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

- [x] Explain search results
- [x] Summarize recommended properties
- [x] Explain ranking decisions

---

## Phase 15.4 — AI Property Analysis

> Combine the existing backend analysis agents and let Claude explain the
> results in natural language.
>
> Existing backend agents remain unchanged. After the existing analysis
> endpoints (POST /analysis/predict, /rental, /valuation, /advisor,
> /negotiation) return their structured result, Claude explains what that
> result means. A thin explanation service (src/llm/analysis_explanation.py)
> reuses the Phase 15.2 Analysis Prompt Builder (build_analysis_prompt) and the
> Phase 15.1 Claude Client (ask_claude) to turn the backend analysis into a
> natural-language explanation. It performs no analysis, prediction, valuation
> or risk scoring, and never invents numbers or recommendations.
>
> The addition is opt-in and additive: each existing /analysis/* endpoint gains
> an optional `explain` query flag (default false → the response is exactly the
> current list, so the API contract is preserved). When `explain=true` the same
> backend rows are returned under `content` plus an optional `ai_explanation`
> (an optional `analysis_type` flag only labels the explanation — e.g. the
> shared /advisor endpoint can be explained as Risk or Advisor). Claude is
> optional: if it fails, `ai_explanation` is null and the backend analysis still
> returns, so an AI failure never blocks the analysis. No new REST endpoint was
> added, no analysis agent, prediction service or ML model was modified.
>
> The frontend reuses the existing AI Analysis page: a single reusable
> AnalysisExplanation card (features/analysis/analysis-explanation.tsx) renders
> the explanation above the existing analysis renderers (AnalysisTable /
> RiskCards / AdvisorCards / NegotiationCards) and shows a graceful "AI
> explanation is temporarily unavailable" message when the backend omits it. The
> analysis service/hook now carry the { content, ai_explanation } response; the
> existing cards, evaluation tray and loading/error states are unchanged. tsc,
> ESLint and next build all pass.

- [x] Explain Risk Analysis — POST /analysis/advisor (analysis_type=risk)
- [x] Explain Rental Analysis — POST /analysis/rental
- [x] Explain Future Growth — POST /analysis/advisor (analysis_type=future; explains the growth_label / growth_reason the backend already produced)
- [x] Explain Valuation — POST /analysis/valuation
- [x] Explain Negotiation Strategy — POST /analysis/negotiation

---

## Phase 15.5 — AI Property Comparison

> Use the existing comparison agent to determine the best property and let
> Claude explain the strengths and weaknesses of each option.
>
> The existing backend comparison agent remains unchanged and is the source of
> truth. After POST /analysis/comparison returns its structured result
> ({ winner, rankings } from compare_properties), Claude explains that result.
> A thin explanation service (src/llm/comparison_explanation.py) reuses the
> Phase 15.2 Comparison Prompt Builder (build_comparison_prompt) and the Phase
> 15.1 Claude Client (ask_claude) to turn the backend comparison into a
> natural-language explanation. It performs no comparison, scoring or ranking,
> and never overrides the winner or invents scores/metrics.
>
> The addition is opt-in and additive: /analysis/comparison gained an optional
> `explain` query flag (default false → the response is exactly the current
> comparison result, so the API contract is preserved). When `explain=true` the
> same backend comparison is returned under `content` plus an optional
> `ai_explanation`. Claude is optional: if it fails, `ai_explanation` is null
> and the backend comparison still returns, so an AI failure never blocks the
> comparison. No new REST endpoint was added, and comparison_agent.py,
> comparison_service.py and comparison_node.py were not modified.
>
> The frontend reuses the existing Property Comparison page: the reusable
> AnalysisExplanation card (features/analysis/analysis-explanation.tsx, now with
> an optional `unavailableMessage`) renders the explanation above the existing
> Property Score Cards and Comparison Table, and shows a graceful "Property
> comparison is available, but the AI explanation is temporarily unavailable"
> message when the backend omits it. The compare service/hook now carry the
> { content, ai_explanation } response; the existing ComparisonResult,
> PropertyScoreCards, evaluation tray and loading/error states are unchanged.
> tsc and ESLint pass.

- [x] Explain comparison results
- [x] Summarize investment advantages
- [x] Generate recommendation summary

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

> The existing backend Investment Advisor agent remains unchanged and is the
> source of truth. When the Investment Advisor is run with `summary=true`, the
> existing `/analysis/advisor` endpoint returns the same advisor rows under
> `content` and additionally gathers the other EXISTING analyses for the same
> properties (price prediction, valuation, rental, negotiation) via the existing
> MCP tools. A thin explanation service (src/llm/investment_explanation.py)
> reuses a new Phase 15.2 prompt builder (build_investment_prompt) and the Phase
> 15.1 Claude Client (ask_claude) to connect those structured results into ONE
> conversational investment summary under `ai_explanation`. It performs no
> prediction, valuation, rental, risk or negotiation analysis, computes no
> investment score, never overrides the backend recommendation, and never
> invents values, risks or opportunities (e.g. Future Growth, which has no
> backend endpoint, is omitted rather than invented).
>
> The addition is opt-in and additive: `/analysis/advisor` gained an optional
> `summary` query flag (default false → the response is exactly the current
> advisor result, so the API contract and the Phase 15.4 `explain`/`analysis_type`
> behavior are preserved). No new REST endpoint was added, and advisor_agent.py,
> analysis_agent.py, risk_agent.py, rental_agent.py, future_agent.py,
> negotiation_agent.py, prediction_service.py and comparison_service.py were not
> modified. Claude is optional: gathering each supporting analysis is defensive
> (a failure degrades to an empty section) and if Claude fails `ai_explanation`
> is null, so the backend Investment Advisor result always returns and never
> breaks.
>
> The frontend reuses the existing AI Analysis page: the Investment Advisor
> button now runs `getInvestmentSummary` (POST /analysis/advisor?summary=true),
> and the reusable AnalysisExplanation card renders the combined investment
> summary above the existing AdvisorCards, with a tailored graceful fallback
> ("Investment analysis is available, but the AI summary is temporarily
> unavailable."). The response shape ({ content, ai_explanation }), AdvisorCards,
> evaluation tray and loading/error states are unchanged; no duplicate components
> were created. tsc and ESLint pass.

- [x] Investment summary
- [x] Pros and Cons
- [x] Final recommendation
- [x] Investment reasoning

---

## Phase 15.7 — Conversational Memory

> Maintain conversation context during a user session.

Claude should remember:

- Previous searches
- Evaluation tray
- Compared properties
- User preferences
- Previous follow-up questions

> Conversational memory is session-scoped, in-memory only — no database and no
> persistent storage. A new module (src/llm/conversation_memory.py) keeps a
> lightweight per-session `ConversationMemory` (recent turns, evaluation-tray
> ids, and the mutable `session_state` dict the EXISTING backend workflow
> already uses for its own follow-up/pagination logic). A thread-safe
> SessionMemoryStore holds these per `session_id`, bounded by a turn cap, a
> session cap and a TTL, and can be cleared. It performs NO business logic:
> no search, ranking, prediction, valuation or recommendation.
>
> The addition is opt-in and additive: POST /chat gained an optional
> `session_id` field (default None → the endpoint behaves exactly as before,
> so the API contract is preserved). When a `session_id` is sent, the endpoint
> keeps that session's `session_state` alive across HTTP requests — which is
> what makes the existing backend follow-up logic (last_search_filters,
> last_search_weights, search_page) work over the stateless API — and passes a
> compact memory summary to Claude as CONTEXT ONLY. Memory is best-effort: if
> it cannot be loaded the endpoint still answers using the current request and
> backend response. No new REST endpoint was added, and chat_service.py, the
> search/recommendation engine, ML models and analysis agents were not
> modified.
>
> The memory summary is threaded to Claude by reusing the Phase 15.2 Search
> Prompt Builder (build_search_prompt gained an optional `memory` context
> block) and the Phase 15.3 explanation service (explain_search_results gained
> an optional `memory` argument). Claude uses memory only to resolve follow-up
> references (e.g. "the cheaper ones", "the previous property") and is
> instructed never to treat memory as backend data or invent remembered facts.
>
> The frontend reuses the existing AI Chat interface with no redesign and no
> new pages: the shared WorkspaceProvider mints a session-scoped `session_id`
> (stable for the provider's lifetime) and sends it with every /chat message.
> Because the provider unmounts on sign-out / navigation, the old session id
> (and therefore its server-side memory) is abandoned when the session ends.
> The ChatRequest type gained the optional `session_id`; the chat UI, evaluation
> tray, property comparison and chat history are unchanged. tsc and ESLint pass.

- [x] Session memory
- [x] Follow-up questions
- [x] Multi-turn conversation

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

> Claude now acts as an intelligent ROUTER on top of the existing chat flow.
> Before the existing keyword workflow runs, Claude selects which EXISTING
> backend capability best matches the user's natural-language message; the
> selected tool is then executed by the existing backend services / MCP tools
> and its unchanged output is wrapped in the response envelope the frontend
> already renders. Claude only routes — it never searches, ranks, predicts,
> values, compares, recommends or generates anything itself.
>
> Selection is a thin, modular layer: a new prompt builder
> (src/llm/prompts/orchestration_prompt.py, reusing the Phase 15.2 `Prompt`
> object and shared APPLICATION CONTEXT) formats a routing prompt from the user
> request, the available backend tools, the evaluation-tray state and the
> Phase 15.7 conversation memory. A new orchestrator
> (src/llm/tool_orchestrator.py) holds the registry of EXISTING capabilities
> and reuses the Phase 15.1 Claude Client (`ask_claude`) to return one selected
> tool (plus internal-only reasoning, or a short `clarify` follow-up). It
> performs NO business logic and does NOT execute the tool.
>
> Execution stays in the existing backend. The existing POST /chat controller
> (src/api/chat_api.py) delegates the selected tool to the existing MCP tools
> (`compare_properties`, `get_price_prediction`, `get_rental_analysis`,
> `get_valuation_analysis`, `get_negotiation_strategy`, `get_investment_advice`,
> `create_property_report`, `send_property_report`) and the existing
> saved-property functions — the SAME functions the existing endpoints use, so
> no backend service or routing logic is duplicated. Search and general chat
> reuse the existing `parse_intent_and_execute` pipeline unchanged.
>
> Fallback is preserved exactly: when Claude is unavailable, unsure, selects
> search/general chat, or fails to parse, the endpoint falls back to the
> existing backend chat behaviour (`parse_intent_and_execute`), so every prior
> phase (15.3 search explanation, 15.4–15.6 analysis/comparison/investment,
> 15.7 memory) keeps working. Tray-based tools return the existing "add
> properties to your tray" prompt when the tray is empty, and report sharing
> asks for a phone number when none is provided — a short clarifying question
> instead of guessing. Backend tool failures surface as the existing HTTP
> errors; Claude never fabricates results.
>
> The frontend is untouched: the orchestration is invisible. Every routed
> response uses a `type` the existing chat renderer already handles
> (search_results / comparison / rental / prediction / valuation / negotiation /
> advisor, and `text` for reports, saved properties and clarifications). No new
> REST endpoint was added, no backend business logic, ML model, recommendation
> or analysis agent was modified, and no API contract changed. Python imports
> and FastAPI startup verified.

- [x] Search tool — routes to the existing search pipeline (`parse_intent_and_execute`)
- [x] Comparison tool — routes to the existing `compare_properties`
- [x] Analysis tools — routes to existing prediction / rental / valuation / negotiation / advisor tools
- [x] Report tools — routes to existing `create_property_report` / `send_property_report`
- [x] Saved-property tools — routes to existing saved-property functions

---

## Phase 15.9 — Streaming Responses

> Stream Claude responses to the frontend for a more responsive chat
> experience.
>
> Streaming is a DELIVERY enhancement only — no backend business logic, AI
> reasoning, or API contract changed. The existing `POST /chat` JSON endpoint is
> byte-identical for non-streaming clients; a thin `POST /chat/stream`
> (Server-Sent Events) is added alongside it and BOTH share one workflow
> (`_run_chat_workflow` in `src/api/chat_api.py`), so no chat logic is
> duplicated. The one reusable Claude Client (Phase 15.1) was extended with a
> `stream()` generator + `stream_claude()` helper (mirroring `generate()`/
> `ask_claude` and their exact error categories); `search_explanation.py` gained
> a `stream_search_explanation()` that reuses the SAME Phase 15.2 Search Prompt
> Builder. In the chat flow the only Claude-generated natural language is the
> search `ai_explanation` (Phase 15.3), so search results stream token-by-token
> then attach the ranked cards on completion, while every other response type
> (comparison / analysis / text) is delivered as a single `done` event after a
> brief `thinking` phase. Business logic runs BEFORE streaming begins so backend
> errors still surface as normal HTTP errors; a failed Claude stream degrades
> gracefully (partial explanation + results still render). Conversational Memory
> (15.7) and Tool Orchestration (15.8) are unchanged — the streaming path runs
> the identical `_run_chat_workflow` and records the assistant turn afterwards.
>
> Frontend reuses the existing AI Chat interface with no redesign: a new
> `streamChatMessage` service reads the SSE stream via `fetch` + a
> `ReadableStream` reader (honoring an `AbortSignal`), and the shared
> `WorkspaceProvider` accumulates deltas into ONLY the active assistant message
> (minimal re-renders), replaces it with the final structured payload on `done`,
> and exposes `stopStreaming` / `isStreaming` / `phase`. The chat message shows a
> blinking typing cursor while streaming; the workspace keeps a thinking
> indicator during the backend phase and auto-scrolls only when the user is near
> the bottom (a manual scroll upward pauses it, returning resumes it). The
> composer's send button becomes a Stop button mid-stream to cancel the active
> stream cleanly (no hanging request). A new send also aborts the previous
> stream. On failure any partial text is kept and the existing Error/Retry
> wiring offers a retry. Rendering stays plain-text `whitespace-pre-wrap`,
> consistent with the existing AI explanation components (no markdown dependency
> added). tsc, ESLint and `next build` all pass; Python imports and FastAPI
> startup verified.

- [x] Streaming API
- [x] Streaming frontend support
- [x] Loading indicators

---

## Phase 15.10 — AI Report Enhancement

> Enhance the existing backend-generated reports by allowing Claude to
> produce professional summaries.

Backend generates the report.

Claude improves readability.

> The existing backend report generator remains unchanged and is the source of
> truth. After POST /report returns its Markdown report (create_property_report),
> Claude improves the readability and structure of that SAME report. A thin
> enhancement service (src/llm/report_enhancement.py) reuses the Phase 15.2
> Report Prompt Builder (build_report_prompt) and the Phase 15.1 Claude Client
> (ask_claude) to re-present the report with an executive summary and clear
> sections (investment, risk, recommendation). It performs NO report generation,
> analysis or calculation, and never invents prices, analysis, recommendations or
> changes any backend conclusion.
>
> The addition is opt-in and additive: POST /report gained an optional `enhance`
> query flag (default false → the response is exactly the current bare Markdown
> string, so the API contract is preserved). When `enhance=true` the same backend
> report is returned under `content` plus an optional `ai_enhanced` (the polished
> version). POST /report/share gained the same optional `enhance` flag so the
> shared report matches the enhanced preview; the existing sharing workflow (MCP
> tool send_property_report + n8n) is otherwise unchanged. Claude is optional: if
> the enhancement fails, `ai_enhanced` is null (and share sends the backend
> report as-is), so an AI failure never blocks report generation or sharing. No
> new REST endpoint was added, and the report generator, analysis agents,
> prediction service, recommendation engine, MCP tools and n8n workflow were not
> modified.
>
> The frontend reuses the existing Reports page and workflow (select → generate →
> preview → download → share) with no redesign: the Report service now requests
> the enhanced report (POST /report?enhance=true), and the reused ReportPreview
> renders the enhanced narrative when available, otherwise the unchanged backend
> report with a friendly "AI enhancement is temporarily unavailable" notice.
> Download always saves the displayed report; the existing loading, error, share
> and download actions are unchanged. tsc and ESLint pass.

- [x] Executive summary
- [x] Investment summary
- [x] Risk summary
- [x] Recommendation summary

---

## Phase 15.11 — AI Suggestions

> Claude suggests useful follow-up actions after every conversation.

Examples:

- Compare these properties
- Generate report
- Analyze rental income
- View property details
- Save property

> After each chat turn, Claude recommends 3-5 short follow-up ACTIONS drawn
> ONLY from the EXISTING EstateMind capabilities — it never invents features,
> executes anything or performs business logic. A thin service
> (`src/llm/suggestions.py`) reuses the Phase 15.2 Prompt Builder (a new
> `build_suggestions_prompt`, registered in `src/llm/prompts`) and the Phase
> 15.1 Claude Client (`ask_claude`) to select next actions from a per-response
> catalog of existing capabilities (search / compare / predict / rental /
> valuation / negotiation / advisor / report / share / save / view saved). The
> catalog is filtered by evaluation-tray state so every suggestion is runnable,
> and the Phase 15.7 conversation memory is passed through as context so
> suggestions stay relevant and avoid repeating the completed action. Claude is
> OPTIONAL: any failure (or non-JSON reply) returns an empty list and the
> suggestion section is simply hidden — the chat response is never affected.
>
> Suggestions are attached to the SAME response envelope in `chat_api.py`
> (`attach_suggestions`) for both `POST /chat` and `POST /chat/stream`. For
> streaming they are computed only AFTER the streamed explanation completes and
> travel in the single `done` payload, so they are never streamed
> token-by-token. No new REST endpoint, backend business logic, ML model,
> recommendation/analysis agent, MCP tool, report generation/sharing or API
> contract was changed — `suggestions` is a purely additive optional field.
>
> The frontend reuses the existing AI Chat interface with no redesign: a new
> reusable `SuggestedActions` component renders the suggestions as
> green-and-white quick-action chips (shared Button) below the completed
> assistant reply, only on the latest assistant message and only once idle.
> Selecting a chip re-sends it through the existing `sendMessage` pipeline, so
> the Phase 15.8 tool orchestration routes it to the correct EXISTING backend
> capability — no new workflow or routing logic is added. Chips wrap gracefully,
> are keyboard navigable and disable while a response is in flight.
> `ChatResponse` / `ChatMessage` gained an optional `suggestions` field. tsc,
> ESLint and `next build` pass; Python imports and the graceful-failure path
> were verified.

- [x] Suggested actions
- [x] Follow-up recommendations

---

## Phase 15.12 — End-to-End Testing

> Verify the complete conversational workflow.
>
> End-to-end verification of every AI feature from Phases 15.1–15.11 was
> performed: `tsc`, ESLint and `next build` all pass clean, and the AI backend
> modules (`src/api/main`, `chat_api`, `claude_client`, `suggestions`,
> `tool_orchestrator`, `conversation_memory`) all import cleanly. The Claude
> layer degrades gracefully everywhere (missing key / timeout / rate limit /
> empty response all return structured errors and never break the backend), and
> memory resets on sign-out because the `WorkspaceProvider` unmounts with the
> protected layout. Three confirmed bugs were found and fixed in the Report
> Sharing workflow (frontend only — no backend, MCP tool or n8n change): the
> "sent successfully" banner now reflects the backend's returned delivery
> `status_code` instead of any HTTP 2xx; the banner shows the phone number
> actually sent (mutation variables) instead of the live input; and Share now
> targets the SAME properties the previewed report was generated from instead of
> the live tray. No new AI features, APIs, workflows or backend logic were
> introduced.

- [x] Search → Claude explanation
- [x] Property Details → Claude explanation
- [x] Comparison → Claude explanation
- [x] Investment Advisor → Claude explanation
- [x] Report Generation → Claude summary
- [x] Report Sharing
- [x] Session memory
- [x] Streaming responses
- [x] Error handling

---

## Phase 15.13 — ChatGPT-style Copilot Workspace

> Redesigned the Copilot into a permanent four-column workspace
> (Conversation Sidebar | Chat + Results | Property Map | Evaluation Tray) and
> made every conversation a complete, restorable EstateMind workspace. This is
> purely frontend state management — no backend, FastAPI endpoint, ML model,
> search/recommendation engine, analysis agent or API contract was changed.
>
> Conversation state (features/dashboard/conversations.ts + rewritten
> workspace-provider.tsx): each conversation stores its own chat messages, the
> ACCUMULATED deduplicated property collection, the evaluation tray and the
> comparison selection, plus a stable id reused as the backend session_id (Phase
> 15.7 memory). Conversations persist to localStorage (client only) so Recent /
> Pinned survive a reload; nothing is sent to the backend. New Chat creates an
> empty workspace; switching a conversation restores its full workspace exactly
> as left; deleting falls back to another conversation (or a fresh one).
>
> Property accumulation: every successful search appends only its NEW properties
> (deduplicated by property id via mergeProperties) to the active conversation —
> 5 + 5 unique -> 10, + 3 -> 13. The Property Results panel
> (property-results-panel.tsx) and the Interactive Property Map (map-panel.tsx)
> both render from that single accumulated collection, so cards and markers never
> replace prior results and never duplicate. The accumulated cards render once,
> under the newest search message; the map auto-fits all markers.
>
> Map ↔ card sync: a shared selectedPropertyId highlights the matching card and
> marker in both directions; clicking a card centres the map on that property
> (new FocusSelected pan in property-map-view.tsx) and scrolls the card into
> view.
>
> Sidebar (conversation-sidebar.tsx + conversation-item.tsx + the new
> dropdown-menu primitive): ChatGPT-style Pinned / Recent sections, New Chat, and
> a per-conversation context menu (Rename inline / Pin or Unpin / Delete). The
> app's global navigation is folded into the sidebar footer so the workspace runs
> full-bleed (DashboardLayout hides the standard chrome on /dashboard and /chat).
> Below xl the map, tray and conversation list open as slide-over sheets so
> nothing overflows on smaller screens. tsc, ESLint and next build all pass; every
> route renders (200) in dev.

- [x] Four-column workspace layout
- [x] Conversation-scoped property accumulation (dedup by id)
- [x] Shared collection for Property Results + Map
- [x] Interactive map: fit bounds, card ↔ marker sync, click-to-centre
- [x] ChatGPT-style Conversation Sidebar (Pinned / Recent)
- [x] Conversation actions (Rename / Pin / Unpin / Delete)
- [x] New Chat clears the workspace; switching restores it
- [x] Premium responsive UI (no backend changes)

---

## Phase 15.14 — Workspace Layout & Navigation

> Frontend-only navigation and UX cleanup. The Dashboard is now the single entry
> point for the Copilot Workspace: the duplicate "AI Chat" sidebar item was
> removed (both routes rendered the same shell). The /chat route still works
> internally for compatibility, and Dashboard stays highlighted anywhere inside
> the workspace (including /chat). The former Map and Tray columns were merged
> into one right column that stacks the Property Map above the Evaluation Tray
> with a draggable divider to resize the map (min 300px, max 85%, tray fills the
> rest; persisted to localStorage). Staged property cards now switch the entire
> card to a premium light-green wash. No backend, API, conversation, map or tray
> logic was changed. tsc and ESLint pass on the changed files.

- [x] Remove duplicate "AI Chat" sidebar item (Dashboard is the single entry)
- [x] Keep /chat route working for compatibility
- [x] Dashboard stays active across the whole workspace
- [x] Right column: Property Map stacked above Evaluation Tray
- [x] Drag-to-resize divider (clamped, smooth, no page scroll) + localStorage
- [x] Premium light-green staged property card styling

---

## Phase 15.15 — Workspace Card & Resize Polish

> Frontend-only visual/layout polish. Staged property cards now switch the whole
> card to a richer (but still elegant) light-green appearance — stronger
> gradient, darker green border, green ring, soft glow and a deeper recommendation
> badge — so staged items are immediately recognizable; unstaged cards are
> unchanged. The workspace split now also resizes horizontally: a draggable
> col-resize divider between the chat column and the right panel, clamped to
> keep the right panel between 25% and 45% of the chat+right region (chat always
> ≥ 55%) and persisted to localStorage; the existing vertical Map/Tray resize is
> untouched and both work together. Property cards in the results list adopt a
> horizontal layout (image left, details right) from the md breakpoint up and
> fall back to the vertical layout on mobile — no information removed, just
> rearranged. No backend, API, conversation, search, map or tray logic changed.
> tsc and ESLint pass on the changed files.

- [x] Stronger premium light-green staged card state (+ deeper recommendation badge)
- [x] Horizontal chat ↔ right-panel resize (25%–45%, persisted)
- [x] Horizontal vertical resize continue working together
- [x] Responsive horizontal property card (md+ horizontal, mobile vertical)

---

## Phase 15.18 — Premium Analysis Workspace UI

> Frontend-only UI enhancement. The AI Analysis page now renders every analysis
> result in the premium report-document design language (Phase 15.17) instead of
> plain tables/cards: solid green hero banners with verbatim chips per property,
> uppercase micro-label metric cards, tone-tinted status pills (shared
> lib/value-tone.ts, extracted verbatim from the report document), dashed
> key-value rows for any extra backend fields (nothing is ever dropped), score
> bars, callout cards (green/red/amber/blue), collapsible static metric
> explainers, and skeleton-card loading states. Dedicated premium renderers for
> Price Prediction (gain/loss difference, current-vs-predicted bars), Valuation
> (three-stop color-coded scale highlighting the backend flag), Rental (rent /
> yield metric grid + strategy callout), Risk, Future Growth (infrastructure &
> signal chips), Advisor (strengths/weaknesses + verdict callout) and
> Negotiation (target price / discount metrics + talking-point checklist). The
> comparison view gets a green winner hero, report-style zebra ranking table and
> score-bar property cards. Shared components (AdvisorCards, NegotiationCards,
> ComparisonResult) keep their signatures, so chat payloads upgrade too. No
> backend, API, response format or analysis logic changed — every displayed
> value comes verbatim from the existing responses. Debug console.log removed
> from the workspace. Production build passes.

- [x] Shared premium primitives (features/analysis/ui/analysis-ui.tsx)
- [x] Premium Price Prediction / Valuation / Rental renderers
- [x] Premium Risk / Future Growth / Advisor / Negotiation renderers
- [x] Premium comparison view (winner hero, score bars, report-style table)
- [x] AI explanation card restyled with the report accent bar
- [x] Skeleton-card loading + elegant empty states

### Density pass — compact investment dashboard

> Frontend-only follow-up. The Analysis page is redesigned for information
> density (~30–40% less scrolling) so it reads like a premium investment
> dashboard rather than a stack of large cards. The tall green hero banner is
> replaced by a compact (~72px) property header (features/analysis/
> property-header.tsx) with a thumbnail from the backend image_urls (elegant
> placeholder otherwise), the property name / location / configuration looked
> up from the workspace's already-loaded search rows (card ids demoted to small
> metadata), a per-analysis eyebrow and a verbatim status pill. Metric cards
> are ~35% shorter (tighter padding, icon + micro-label typography) and laid
> out in dense 2-up / 4-up grids; the long full-width bars are replaced by
> mini comparison bars (prediction), a radial yield/score gauge (rental,
> growth), a compact segmented indicator (valuation) and a difference badge
> with direction arrow (prediction). Risk/Advisor/Negotiation callouts pair up
> in 2-column grids; the comparison winner banner and score cards are
> flattened to slim strips. The AI insight card ("EstateMind Insight") is
> compact, and a missing explanation renders as a small dashed empty state
> instead of a large banner. All KeyValueList extra-field rendering is kept so
> no backend field is ever dropped; no backend, API, response format or
> analysis logic changed. Production build passes.

- [x] Compact property header (name/location/config + thumbnail, id as metadata)
- [x] Compact metric cards (-35% height) in dense 2x2 / 4-up grids
- [x] Mini comparison bars + difference badge (prediction)
- [x] Radial gauges (rental yield, growth score) + segmented valuation indicator
- [x] Paired callout grids (risk, advisor, negotiation)
- [x] Slim comparison winner strip + compact score cards
- [x] Compact EstateMind Insight card + elegant unavailable state

---

## Phase 15.20 — Analysis Result Clarity & Decision-Oriented UX

> Frontend-only presentation change. Every analysis now reads decision-first,
> in the same executive-report hierarchy: Decision Summary → Why this result?
> → Metrics → Recommendation → collapsed Technical details. New shared
> primitives (features/analysis/ui/decision-summary.tsx): DecisionSummary (a
> large tone-tinted headline card with an action tagline and an optional
> prominent stat), WhyCard (✓ checklist assembled from backend fields),
> RecommendationBar (reuses CalloutCard) and TechnicalDetails (collapsible
> KeyValueList container); lib/value-tone.ts gains toneKey() so the summary
> cards and status pills share one wording→tone rule. Per analysis: Price
> Prediction derives Overpriced / Undervalued / Fairly Priced from the SIGN of
> the backend margin_diff (the same reading the old difference badge colored)
> with the predicted value as the headline stat; Rental leads with the verbatim
> investment_rating (restated as stars) and the verbatim yield, closing with
> the backend rental_strategy; Valuation leads with the verbatim analysis_flag
> and closes with the verbatim analysis_msg; Risk leads with the verbatim
> verdict plus a concern count from the backend risks list; Future Growth leads
> with the verbatim growth_label and uses the infrastructure/signal lists as
> the Why checklist; Advisor leads with the verbatim verdict and closes with
> the verbatim suitable_for fit; Negotiation leads with the verbatim
> negotiation_power and the backend target price as the headline stat. The
> comparison winner strip shows the project name (workspace lookup, id demoted
> to metadata). Duplicate status displays removed: the property-header status
> pill, the rating/verdict/growth/power pill cards and the valuation segmented
> indicator that repeated the summary are gone — each major status appears
> exactly once. Extra backend fields stay reachable via the collapsed
> Technical details (KeyValueList unchanged, nothing dropped). No backend,
> API, response format or analysis logic changed; every headline, reason and
> action is derived only from existing response fields. Shared components
> (AdvisorCards, NegotiationCards, ComparisonResult) keep their signatures, so
> chat payloads upgrade too. tsc, ESLint and the production build pass.

- [x] Shared decision-first primitives (DecisionSummary / WhyCard / RecommendationBar / TechnicalDetails)
- [x] toneKey() shared wording→tone rule (lib/value-tone.ts)
- [x] Decision-first Price Prediction / Rental / Valuation renderers
- [x] Decision-first Risk / Future Growth / Advisor / Negotiation renderers
- [x] Comparison winner strip shows the project name
- [x] Duplicate status displays removed (status shown exactly once per analysis)
- [x] Extra backend fields collapsed into Technical details (nothing dropped)

---

## Phase 15.21 — Cross-Property Executive Comparison UX

> Frontend-only presentation change. When several properties are analyzed
> together, every analysis now opens with an Executive Comparison Summary —
> "which property performs better in this category?" — before the unchanged
> Phase 15.20 per-property sections. New reusable component
> (features/analysis/ui/executive-summary.tsx): a Premium Report green winner
> banner (trophy eyebrow, large property name resolved from the workspace's
> loaded search rows, verbatim status badge, star restatement, one-line
> explanation, key stat, ✓ reasons) over a side-by-side contender strip
> showing every compared property with its verbatim status and value; a
> "Property breakdown" section label separates it from the individual
> sections. The winner is never computed — each renderer picks it by comparing
> EXISTING backend values only: Price Prediction takes the largest backend
> margin_diff (asking furthest below prediction, or closest to it when all are
> overpriced); Rental the highest backend rental_yield_percent; Valuation the
> best backend analysis_flag on the backend's own scale (undervalued > fair >
> overpriced); Risk the best verdict wording (shared toneRank in
> lib/value-tone.ts), ties to the fewest backend-listed concerns; Future
> Growth the highest backend growth_score (falling back to the growth_label
> wording); Advisor the best-graded verdict (star restatement, tone as
> tie-break); Negotiation the largest backend suggested discount (power
> wording as tie-break). Summaries render only when 2+ properties carry the
> needed backend values, so single-property analyses are unchanged. The
> comparison view is polished into winner + runner-up cards: a larger green
> winner card whose comparison_reason splits into ✓ advantages, and a
> runner-up card (next backend-ranked property) with score, verdict pill and
> reason. No backend, API, response format or analysis logic changed; shared
> component signatures unchanged, so chat payloads upgrade too. tsc, ESLint
> and the production build pass.

- [x] Reusable ExecutiveSummary (winner banner + contender strip + usePropertyName)
- [x] toneRank() ordinal wording comparison + StarRow onDark variant
- [x] Executive comparison on all seven analyses (backend-value picks only)
- [x] Single-property analyses unchanged (summary gated on 2+ contenders)
- [x] Comparison polish: large winner card with ✓ advantages + runner-up card
- [x] Phase 15.20 per-property hierarchy fully preserved below the summary

---

## Phase 16.1 — WhatsApp Cloud API Report Delivery

> End-to-end WhatsApp report delivery through the official Meta WhatsApp
> Cloud API — no n8n, Twilio, pywhatkit or Selenium. New reusable backend
> service (src/services/whatsapp_service.py): validate_phone_number()
> (normalize to digits-only international format, reject invalid input),
> generate_report_pdf() (renders the EXISTING Markdown report text verbatim
> to a branded A4 PDF with reportlab — headings, bullets, emphasis — cached
> by content hash so each report is rendered once), upload_pdf() (POST
> /{PHONE_NUMBER_ID}/media multipart → media_id), send_document() (POST
> /{PHONE_NUMBER_ID}/messages document message) and send_report() (the
> orchestration, returning the frontend-compatible ShareResult {status,
> status_code, message_id}). Credentials come only from environment
> variables (WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID,
> WHATSAPP_API_VERSION=v25.0 — documented in .env.example, loaded via
> python-dotenv, read at call time). Meaningful errors: invalid phone → 400,
> missing config → 503, expired/invalid token (OAuthException) → 502 with a
> clear message, upload/send failures → 502 with Meta's own error, network
> timeout → 504, missing/empty report or PDF → 400/500. POST /report/share
> now delivers via this service and accepts an optional `report` field: the
> frontend passes the previewed report text so the exact on-screen report is
> sent WITHOUT regenerating it (when absent, the report is generated once —
> the pre-16.1 contract). Frontend keeps the existing UI: same phone textbox
> and Share Report button, "Sending report…" pending state, "Report sent
> successfully" confirmation and an "Unable to send report" error state that
> surfaces the backend's meaningful error detail. Report generation, prompts,
> analysis and layout untouched. Verified with mocked Cloud API end-to-end
> tests (upload → send payloads, token expiry, timeout), real PDF generation,
> FastAPI app import, tsc/ESLint and the production build.

- [x] src/services/whatsapp_service.py (upload_pdf / send_document / send_report)
- [x] Branded PDF from the existing report text (reportlab, content-hash cache)
- [x] Env-only credentials (.env.example updated, no hardcoding)
- [x] POST /report/share switched to WhatsApp Cloud API + optional report passthrough
- [x] Frontend sends the previewed report text (never regenerated)
- [x] Meaningful error handling (invalid phone / expired token / upload / timeout / missing PDF)
- [x] Existing Share Report UI preserved (wording aligned: sending / success / failure)

---

## Phase 16.2 — Pixel-Identical PDF Pipeline (Chromium)

> ONE PDF generator for the whole application. The ReportLab renderer is
> gone; src/services/pdf_service.py now renders the EXISTING Next.js report
> document with headless Chromium (Playwright for Python — the backend is
> Python, so no Node sidecar) and exports it as an A4 PDF. New print-only
> route frontend/app/print/report/page.tsx reuses the approved
> ReportDocument + report-parser + the existing .report-print-root print CSS
> untouched (portaled to <body> exactly like the Reports page print copy),
> so the PDF preserves the EstateMind header, green property banners,
> two-column card grid, comparison tables, badges, stars, rounded corners,
> shadows, page-break rules and the repeating footer — pixel-identical to
> the browser's Export PDF / Chrome print output. The renderer injects the
> previewed report text as window.__ESTATEMIND_REPORT__ (nothing is
> regenerated), waits for network idle + fonts + the data-report-ready flag,
> then prints (A4 portrait, printBackground, preferCSSPageSize; margins from
> the existing @page rule). Results cached by content hash. WhatsApp
> delivery (whatsapp_service.py) keeps the same Cloud API upload/send code
> but ships this PDF; new thin POST /report/pdf endpoint serves the same
> file, and the Export PDF button downloads it (falling back to browser
> print if the renderer is unavailable) — so Export PDF, Download and
> WhatsApp always deliver the identical document. Requires Playwright
> Chromium (playwright install chromium) and the Next.js frontend running
> (FRONTEND_BASE_URL, default http://localhost:3000 — documented in
> .env.example). Report content, workflows and all report UI unchanged.
> Verified end-to-end: Chromium render of the print route, PDF content
> probes (header/cards/badges/footer), POST /report/pdf (200 +
> application/pdf, 400 on empty report), FastAPI import without reportlab,
> and the production build with the new /print/report route.

- [x] frontend/app/print/report/page.tsx (print-only route, reuses ReportDocument — no duplicate layout)
- [x] src/services/pdf_service.py (single Chromium PDF generator, content-hash cache)
- [x] whatsapp_service.py delivers the Chromium PDF (ReportLab removed; Cloud API untouched)
- [x] POST /report/pdf (thin endpoint, same generator)
- [x] Export PDF button downloads the same PDF (window.print fallback only)
- [x] requirements: reportlab → playwright; FRONTEND_BASE_URL in .env.example

---

## Phase 17.0 — Advanced Property Comparison Workspace

> New dedicated /compare page (left navigation: "Property Comparison") for
> professional side-by-side comparison of 2–3 properties staged in the shared
> Evaluation Tray. 100% frontend presentation and orchestration: one flow
> fetches the EXISTING backend results — POST /analysis/comparison first (it
> warms the backend's per-property enrichment cache and provides the winner +
> rankings), then the raw analysis rows (predict / rental / valuation /
> advisor / negotiation, called without `explain` so no unused AI text is
> generated) and GET /property/{id} in parallel. Page flow: property selector
> (three dropdown slots, min 2 / max 3, Show Comparison enables at 2) →
> Executive Winner Summary (the verbatim backend comparison winner + runner-up
> + Claude's optional explanation) → Comparison Matrix (every backend property
> field row-wise, unknown fields under Additional Information — never dropped)
> → seven AI analysis comparison sections with Section Winner strips → Final
> Scoreboard → Final Recommendation (backend winner; the closing action is the
> winner's verbatim backend negotiation strategy). Smart highlighting only
> compares existing backend values (soft emerald best / soft rose worst, via
> the shared toneRank for status wording — the Phase 15.21 rule); trophy
> badges mark winning cells; metric labels carry ⓘ tooltips (new shared
> radix-ui Tooltip). Sticky attribute column + sticky property headers, zebra
> rows, entrance animation when switching compared properties. No backend
> changes; the AI Analysis compare card remains.

- [x] Navigation entry + /compare route (standard chrome)
- [x] Property selector fed by the Evaluation Tray (min 2 / max 3)
- [x] use-compare-data orchestration over existing endpoints only
- [x] Executive Winner Summary + Runner Up + AI explanation
- [x] Side-by-side Comparison Matrix (all backend fields, smart highlighting)
- [x] AI analysis comparison sections (prediction / rental / risk / growth / valuation / advisor / negotiation) + section winners
- [x] Final Scoreboard + Final Recommendation
- [x] Shared Tooltip component (components/ui/tooltip.tsx)

---

## Phase 17.1 — Premium Comparison Experience & Decision Dashboard

> Frontend-only polish of the /compare page into a premium investor decision
> dashboard — no backend, API, prompt or business-logic changes; every number
> shown is an existing backend value. Overall Winner Scoreboard right after
> the Executive Summary (win-tally counts the EXISTING Phase 17.0 category
> winners per property — medals, animated proportional win bars, category
> chips; pure counting, no new scoring). Numeric matrix rows gain proportional
> value bars (best value = fullest green bar) and relative-difference helper
> lines ("₹1.05 Cr cheaper than …", "+4.04% higher yield than …" — display
> arithmetic on backend values). Property Overview columns are headed by
> premium mini cards (image, name, id, price, configuration, area, location,
> advisor verdict pill + stars, "🏆 Overall Winner" badge; winner column gets
> a subtle green tint). Basic Information + all seven analysis sections become
> one accordion (new shared radix-ui Accordion, tw-animate-css expand/collapse,
> open sections persisted to localStorage). Section winner strips became "Why
> this property won" cards with an explicit Why? ✓ checklist (richer
> backend-derived points from compare-winners — no AI call). Export Comparison
> PDF + Share on WhatsApp reuse the EXISTING report pipeline verbatim
> (POST /report → /report/pdf → /report/share; report generated once and
> reused for both). Final Recommendation upgraded into the Final Decision
> Card: backend overall score + "Won X of Y categories" stat (counted, never
> invented — no fabricated confidence %), Why checklist, verbatim backend
> negotiation strategy as the recommended action. Hover elevation and smooth
> transitions throughout; sticky headers/attribute column and mobile
> horizontal scroll retained.

- [x] Shared Accordion component (components/ui/accordion.tsx)
- [x] Overall Winner Scoreboard + win-tally (counting existing category winners only)
- [x] Proportional value bars on numeric comparison rows
- [x] Relative-difference helper lines (frontend display arithmetic only)
- [x] Premium property header cards + overall-winner column tint
- [x] Collapsible comparison sections with persisted open state
- [x] "Why this property won" cards after every section
- [x] Export Comparison PDF + Share on WhatsApp via the existing report pipeline
- [x] Final Decision Card (backend score + category-wins stat, no invented confidence)
- [x] Micro-interactions (hover elevation, animated bars, smooth accordion)

---

## Phase 17.2 — Comparison UX Polish & 5-Second Decisions

> Frontend-only polish of the /compare page so the decision is readable in
> seconds — no backend, API, prompt or logic changes. At-a-Glance winner strip
> right below the Executive Summary (Lowest Price / Highest Rental / Lowest
> Risk / Best Growth / Best Investment — the EXISTING compare-winners category
> winners with their backend-derived reason + supporting metric). "Show only
> differences" toggle (persisted to localStorage) hides rows whose backend
> values are identical across every compared property, with a per-table hidden
> count and an all-identical empty note. Accordion sections open with premium
> banners (icon chip + title + one-line description). Property header cards
> gain quick actions reusing EXISTING pages — Open Property Details
> (/property/{id}), Run Individual Analysis (/analysis), Generate Report
> (/reports; the compared properties are already staged in the shared tray) —
> plus Remove From Comparison (re-runs the comparison on the remaining ids, or
> clears the result below 2). Floating compact winner card (name, verdict
> stars, category wins, price — all already-displayed values) appears
> bottom-right once the Executive Summary scrolls out of view
> (IntersectionObserver), dismissible. Export toolbar extended with Print
> Comparison (browser dialog) and Full Report (navigates to the existing
> Reports page) alongside the Phase 17.1 PDF/WhatsApp actions. Hover polish:
> image zoom on header cards, gently pulsing Overall Winner ribbon, soft
> elevation everywhere. Risk section gains a "fewer risk indicators than …"
> relative note (display arithmetic only).

- [x] At-a-Glance winner strip (existing category winners, reason + metric)
- [x] Show Only Differences toggle (persisted, per-table hidden-row count)
- [x] Premium accordion section banners (icon chip + description)
- [x] Property header quick actions (details / analysis / report / remove)
- [x] Remove From Comparison re-runs on the remaining properties
- [x] Floating winner card on scroll (IntersectionObserver, dismissible)
- [x] Print Comparison + Full Report export actions
- [x] Hover polish (image zoom, pulsing winner ribbon)

---

## Phase 17.3 — Comparison Report & WhatsApp Integration

> The Comparison page (/compare) now exports and shares a dedicated
> COMPARISON REPORT that mirrors the comparison view itself — Executive
> Summary, Best Overall Investment, Compared Properties, Category Winners
> (the Winner Strip), side-by-side comparison tables (overview, price
> prediction, rental, risk, future growth, negotiation, scores & verdicts),
> Negotiation Insights and the Final Recommendation — instead of the standard
> per-property investment report. New thin endpoints POST /report/comparison
> and POST /report/comparison/share reuse the whole EXISTING pipeline (the
> backend comparison + analyses via the Phase 15.10 _gather_analyses, the
> Claude client, the single Chromium PDF generator and the WhatsApp Cloud API
> delivery with a comparison filename) with a new comparison report prompt
> template (src/llm/prompts/comparison_report_prompt.py) that follows the
> same plain-text divider/icon conventions, so the existing print route and
> PDF pipeline render it unchanged. If the Claude presentation is
> unavailable, the endpoints degrade to the standard report so sharing never
> hard-fails. The Reports page (/reports) and its /report, /report/pdf and
> /report/share endpoints are COMPLETELY UNCHANGED — the two flows are fully
> independent.

- [x] Comparison report prompt template (comparison-view layout, plain text)
- [x] Comparison report generator reusing existing backend analyses
- [x] POST /report/comparison + POST /report/comparison/share (thin endpoints)
- [x] /compare export toolbar uses the comparison endpoints (PDF + WhatsApp)
- [x] Reports page flow untouched (independent, backward compatible)

---

## Phase 17.4 — Comparison Report Matches the Premium Comparison Dashboard

> The Comparison Report is now the /compare page printed into a premium PDF
> instead of a text report. The comparison report prompt template
> (src/llm/prompts/comparison_report_prompt.py) was redesigned to emit the
> compare page's own structure — Best Overall Investment hero, Overall
> Comparison Score (win tally), ONE side-by-side '|' table per analysis
> (Basic Information, Price Prediction, Rental, Risk, Future Growth,
> Valuation, Investment Advisor, Negotiation) with '🏆 ' prefixes on the
> winning cells and a per-table Winner line, then the Final Scoreboard and
> Final Recommendation. A new frontend renderer
> (features/reports/comparison-report-document.tsx) renders that report with
> the compare page's own visual system: the green winner hero with score /
> category-wins stats, proportional win bars with category chips, emerald 🏆
> winner cells in every table, per-section winner strips, the Final
> Scoreboard card grid and the large green Final Recommendation block. The
> print route (/print/report) picks this renderer only when the parsed
> report title is the comparison title (isComparisonReport), so the standard
> Reports page document, parser, prompt, PDF and WhatsApp flows are
> COMPLETELY UNCHANGED — same single Chromium PDF pipeline, same endpoints,
> nothing regenerated.

- [x] Comparison prompt emits the compare-page layout (side-by-side tables, 🏆 winner cells, win tally, scoreboard)
- [x] ComparisonReportDocument renderer (winner hero, win bars, emerald winner cells, scoreboard cards, recommendation block)
- [x] /print/report routes comparison reports to the new renderer (standard path untouched)
- [x] Verified end-to-end through the real Chromium PDF pipeline
- [x] Reports page flow untouched (independent, backward compatible)

---

## Phase 17.5 — Global Number Formatting & Instant Loading Feedback

> UI polish only — no backend, API or business-logic changes. ONE shared
> formatting utility (features/dashboard/format.ts: formatScore, formatPercent,
> formatInteger, formatCurrency, formatNumber, and a number-aware formatCell)
> now cleans every backend numeric display, so raw floating-point precision
> (0.7000000000000001) never reaches the UI: scores always show two decimals,
> percentages show two decimals, counts stay integers, currency/area formatting
> unchanged. Applied across the dashboard comparison result, all /analysis
> renderers, the /compare workspace (executive summary, score cards, matrix
> tables, winners, final recommendation), property details ratings, property
> cards, and the report preview/print documents (report-parser rounds float
> artifacts — long runs of 0s/9s — to two decimals; verbatim otherwise).
> WhatsApp sharing feels instant: the comparison share now generates the report
> INSIDE the mutation so "Sending…" + spinner appear in the same render frame
> as the click (previously nothing changed for 20-30s), phone inputs and share
> buttons are disabled while sending, a grey "Sending … report to WhatsApp…"
> helper line shows under the form, success banners restate the delivered
> number, and failures show "Unable to send report" + the backend reason with
> a Try Again button. Export PDF buttons show the shared spinner. Same
> endpoints, same order — presentation only.

- [x] Shared number formatters (formatScore / formatPercent / formatInteger / formatCurrency / formatNumber)
- [x] Formatting applied across dashboard, /analysis, /compare, property details and property cards
- [x] Report preview/print float-artifact cleanup (parser-level, presentation only)
- [x] Instant "Sending…" state for WhatsApp share (property + comparison reports)
- [x] Duplicate-click protection (send button, phone input, share toggle disabled while sending)
- [x] Sending helper text, improved success (delivered-to number) and error (reason + Try Again) feedback

## Phase 18.1 — Premium SaaS UI Redesign

> Visual-only redesign to production-quality SaaS standards (Stripe / Linear /
> Vercel-inspired, emerald-green identity). No feature, routing, API or
> business-logic changes — every page keeps its exact workflow. Design tokens
> refreshed in globals.css: soft slate canvas (#F8FAFC), pure-white floating
> cards, brand palette #15803D / #22C55E / #4ADE80, thin #E5E7EB borders,
> #F0FDF4 hover wash, larger radius scale, layered soft shadows (shadow-float /
> shadow-float-lg), glass utility, brand gradient utility, gradient top-accent
> utility, shimmer skeletons and a global prefers-reduced-motion fallback.
> Chrome floats: detached rounded glass sidebar + sticky glass top bar (rounded
> global search with ⌘K hint, notifications, real user-initials avatar linking
> to Profile). The Copilot workspace columns (conversations / chat / map /
> tray) render as floating panels with transparent gutter drag handles —
> resize + streaming logic untouched. Chat got gradient/glassy avatars,
> premium bubbles and a floating rounded composer. The evaluation tray rows
> now show the property thumbnail, name, price and AI recommendation score
> (looked up from the conversation's already-accumulated backend results —
> nothing fetched or computed) plus a selection progress bar. Property cards
> lift on hover with a slow image zoom, gradient BHK badge and refined price
> typography. Buttons: brand-gradient primary with hover lift + soft green
> shadow. Badges are pills; empty states get a soft gradient icon ring; auth
> pages get a soft emerald glow canvas; page headers upgraded to a larger
> heading scale. Print/PDF report documents intentionally untouched.

- [x] Design tokens + premium utilities (glass, shadow-float, brand gradient, shimmer, reduced motion)
- [x] Floating glass sidebar + sticky glass top bar with rounded search and avatar
- [x] Copilot workspace as floating panels (conversations / chat / map / tray)
- [x] Premium chat bubbles, avatars and floating composer
- [x] Evaluation tray: thumbnails, name, price, AI score, progress section
- [x] Premium property card (hover lift, image zoom, gradient badge, pill amenities)
- [x] Gradient buttons, pill badges, shimmer skeletons, premium empty states
- [x] Auth pages glow canvas + larger page-header typography
- [x] Production build verified (next build passes)

---

## Phase 18.2 — Premium Visual Theme Upgrade (Purple, Reference-Based)

> Visual-only retheme of the entire application to the premium purple-and-white
> design language of the provided reference (Linear / Stripe / Perplexity
> feel). No feature, routing, API or business-logic changes. Design tokens
> swapped in globals.css: soft lavender canvas (#F6F5FD), brand purple
> #6D4AFF / #8C6DFF, #ECE9F8 borders, #F3F0FF hover wash, #F5F2FF lavender
> chip surfaces, purple chart scale, purple-tinted float shadows and new
> bg-sidebar-gradient + shadow-brand-glow utilities. Both sidebars (global
> rail and Copilot conversation rail) are now dark purple gradient floating
> panels (#221A44 → #18132F) with light text, purple-gradient active pill with
> soft glow, translucent hover states, dark search input and a bottom user
> profile card with Pro badge and settings glyph (still one link to
> /profile). Evaluation tray rows redesigned per spec: thumbnails removed
> entirely (no placeholder / broken-image icon) — each row shows the cardid in
> 11px gray monospace above the bold name, bold price, green AI score and a
> right-aligned delete icon; selected rows get a purple border and lavender
> wash. Compare/clear/selection behavior untouched. Property cards: bold
> purple project name, purple AI Recommendation badge, lavender amenity
> pills. Comparison matrix headers use the lavender accent (winner columns
> keep their soft green highlight); success/warning/danger semantics remain
> green/amber/red throughout. Map price markers and cluster bubbles inherit
> the purple tokens. Print/PDF report documents intentionally untouched.

- [x] Purple design tokens (canvas, brand, borders, hover, charts, dark mode)
- [x] Dark purple gradient sidebars (global + conversation rail, mobile sheets)
- [x] Sidebar user profile card with Pro badge + settings glyph
- [x] Evaluation tray rows: cardid + name + price + green AI score, no thumbnails
- [x] Purple gradient buttons / glow shadows (button, brand, chat avatar, auth)
- [x] Property card: purple name, purple AI badge, lavender amenity pills
- [x] Comparison matrix: lavender accent headers (green winner highlights kept)
- [x] Production build verified (next build passes)

---

## Phase 18.3 — Dashboard & Property Cards Premium Polish

> Visual-only polish of the Dashboard workspace and the reusable Property
> Card. No feature, routing, API, state-management or business-logic changes.
> Property card: ~25% taller imagery (aspect 16/13) with hover zoom, a subtle
> bottom dark gradient for badge readability, an image-count badge, and a
> premium lavender-gradient "Image Unavailable" placeholder (building icon —
> never a broken-image glyph or empty white box). Typography hierarchy
> re-weighted: dominant bold 2xl price, larger bold purple project name,
> small gray monospace cardid, muted icon-aligned location, evenly-spaced
> Lucide spec row (beds / baths / balconies / parking / area). AI
> recommendation badge redesigned as a purple gradient pill ("✨ AI
> Recommended · Score …", backend value only); amenity chips as padded
> lavender pills with a hover wash; description at 3-line clamp with relaxed
> leading. Actions row: gradient "View Listing" primary CTA beside an
> outlined purple "Read More", consistent heights; the Staged state renders a
> soft green pill. Whole card lifts slightly with a deeper soft shadow on
> hover (150–200ms). Chat composer: more generous padding, softer
> placeholder, purple focus ring + shadow lift. Map panel header: purple icon
> chip beside the title; map card gets the float shadow. Evaluation tray and
> skeletons keep their Phase 18.2 design.

- [x] Property image: taller frame, hover zoom, gradient overlay, photo-count badge
- [x] Premium lavender "Image Unavailable" placeholder (building icon)
- [x] Typography hierarchy: dominant price → purple name → mono cardid → location → specs
- [x] Purple gradient AI Recommended pill (backend score only)
- [x] Lavender amenity pills with hover wash; 3-line clamped description
- [x] Gradient View Listing CTA + outlined purple Read More; green Staged pill
- [x] Card hover lift + deeper shadow (150–200ms transitions)
- [x] Composer, map panel header and spacing polish
- [x] Production build verified (next build passes)

---

## Phase 18.4 — Premium Theme Customization

> Visual-only theme customization. Four built-in premium themes — Estate
> Green (default), Royal Purple (the Phase 18.2 palette), Midnight Blue and
> Sunset Gold — defined entirely as design-token sets in globals.css
> (`:root` + `[data-theme="…"]` overrides, incl. variable-driven sidebar
> gradient and brand glows). A ThemeProvider applies the selection as
> `data-theme` on <html>, persists it to localStorage
> (`estatemind.theme`) and cross-fades colors (~200ms) on switch; an inline
> boot script in the root layout restores the theme before hydration, so
> there is no flash of the default. Profile gains an "Appearance" section
> with four accessible radio-group theme cards (swatches, name, description,
> check indicator, hover lift). No layout, routing, API, state-management or
> business-logic changes.

- [x] Four theme token sets in globals.css (Estate Green default)
- [x] ThemeProvider + localStorage persistence + no-flicker boot script
- [x] ~200ms cross-fade on theme switch (reduced-motion honored)
- [x] Profile → Appearance section with premium theme cards
- [x] Production build verified (next build passes)

---

## Phase 18.6 — Property Comparison Premium Polish

> Presentation-only redesign of the /compare workspace in the Phase 18.5
> design language. Final Recommendation now uses the brand-gradient hero
> with radial sheen, larger stats and icon-chip negotiation footer; the
> runner-up strip, Final Scoreboard and Overall Scoreboard become floating
> accent-top cards with icon chips, tinted medal chips, gradient leader
> bars and bordered category pills. Property header cards get a hover lift
> plus a soft emerald winner border/glow and emerald winner ribbon; winning
> matrix cells gain a thin inset emerald ring and bordered trophy badges;
> rows use softer zebra/hover tones. Winner-strip cards, section winner
> cards and the compare selector adopt accent-top/gradient card chrome with
> hover lift. Export toolbar wrapped in a floating card with button hover
> lifts. Empty states (empty tray incl. a Search Properties CTA, pick your
> contenders) render on soft gradient cards; loading becomes an
> "AI is comparing" sparkle banner over layout-shaped shimmer skeletons.
> The floating winner card glows emerald. No backend, API, routing,
> comparison-logic, export or WhatsApp changes — same components, same
> data, same behavior.

- [x] Final Recommendation on brand-gradient hero + icon-chip action footer
- [x] Scoreboards as floating accent-top cards (medal chips, gradient bars)
- [x] Emerald winner treatment: header-card glow/ribbon, inset-ring winning cells
- [x] Winner strip, section winner cards, selector and toolbar hover polish
- [x] Premium empty states with CTA + AI-comparing shimmer skeleton
- [x] Production build verified (next build passes)

---

## Phase 18.7 — Reports, Saved Properties & Profile Premium Polish

> Presentation-only redesign of /reports, /saved and /profile pages to match
> the premium quality established in Phases 18.3–18.6. Reports workspace gets
> a hero header with gradient icon, refined toolbar with gradient backgrounds,
> premium preview container and enhanced share form. Saved Properties becomes
> a premium collection with hero header, gradient icon, property count display
> and improved empty state with Browse CTA. Profile becomes a premium account
> dashboard with large gradient avatar header, membership badges, account stats
> cards (Reports, Conversations, Status), refined section cards with gradient
> icon containers, premium report cards with hover lift, enhanced user info
> card and polished appearance selector with larger swatches. Empty states
> across all pages get enhanced gradients and better spacing. No backend, API,
> routing, business logic, report generation, PDF, WhatsApp, authentication or
> state management changes — same data, same behavior, premium presentation.

- [x] Reports workspace: hero header, refined toolbar, premium preview/share
- [x] Saved Properties: hero header, collection layout, enhanced empty state
- [x] Profile: gradient avatar hero, stats cards, refined sections
- [x] Premium report cards with hover effects and better visual hierarchy
- [x] Enhanced empty states with improved gradients and spacing
- [x] Polished appearance section with larger theme cards
- [x] All pages match Dashboard/Analysis/Comparison premium quality

---

## Phase 18.8 — Production Quality Polish & Final UX Audit

> Final UI phase: a complete production-quality audit and polish pass across
> the whole application — not a redesign, no feature/routing/API/business-logic
> changes. Audited every page (Dashboard, AI Analysis, Comparison, Reports,
> Saved, Profile, Login, Signup, Landing), the shared UI primitives (button,
> badge, card, input, textarea, skeleton, empty/error states), layout chrome
> (sidebar, navbar, mobile nav), dialogs/drawers/tooltips, tables, forms,
> loading/empty/error/success states, animations (150–200ms standard),
> responsiveness and accessibility (focus rings, ARIA labels, keyboard nav,
> reduced motion — all verified in place from prior phases).
>
> Consistency fixes applied: the Profile account-stats cards no longer use
> hardcoded emerald/blue palette colors — all three stat cards now read the
> theme's primary tokens so they recolor correctly under all four themes
> (Estate Green, Royal Purple, Midnight Blue, Sunset Gold); the comparison
> export toolbar's PDF-failure message uses the semantic `text-destructive`
> token instead of a hardcoded red; the Overall Scoreboard's silver-medal chip
> uses `bg-muted` tokens instead of slate; the CalloutCard "blue" info tone
> (used by neutral RecommendationBars) swapped its hardcoded blue palette for
> theme `muted`/`border` tokens so it no longer clashes with (or vanishes
> into) the Midnight Blue theme; the Saved Properties empty-state container
> dropped its non-standard 2px dashed border for the standard 1px. Semantic
> status colors (emerald = success/winner, amber = warning, red = destructive)
> are intentionally kept hardcoded across themes, matching the Phase 18.2 rule.
>
> Code-quality fixes surfaced by the audit: resolved a duplicate `User`
> identifier (Lucide icon vs. profile type) in user-info-card.tsx; fixed all
> four outstanding ESLint `react-hooks` errors — the theme-provider and
> report-preview hydration-sync effects are documented and scoped, and the two
> dynamically-resolved Lucide icons (conversation rows, comparison-report
> sections) now render via `createElement` so no component is created during
> render. ESLint now passes with zero errors and zero warnings.
>
> Final validation: TypeScript (`tsc --noEmit`) passes, ESLint passes clean,
> and the production build (`next build`) succeeds with every route generated
> (/, /login, /signup, /dashboard, /chat, /analysis, /compare, /reports,
> /saved, /profile, /property/[id], /print/report). Print/PDF report documents
> intentionally untouched. No backend, API, routing, state-management or
> business-logic changes — same data, same behavior, production presentation.

- [x] Global design audit (spacing, typography, radius, shadows, hover, alignment)
- [x] Theme-token audit: removed hardcoded palette colors from Profile stats, comparison toolbar, scoreboard medals, info callouts
- [x] Button / form / card / table / dialog / drawer audits (shared primitives verified consistent)
- [x] Empty / loading / error / success state audits (shared components verified consistent)
- [x] Animation audit (150–200ms, reduced-motion honored globally)
- [x] Accessibility audit (focus-visible rings, ARIA labels, keyboard nav, radiogroup themes)
- [x] Fixed duplicate `User` identifier in user-info-card.tsx
- [x] Fixed all 4 ESLint react-hooks errors (zero errors, zero warnings)
- [x] TypeScript passes · ESLint passes · production build generates all routes
- [x] Theme switching verified token-driven across every audited component

---

## Phase 18.9 — Final UX Polish, Navigation & Persistence

> Production-readiness pass fixing the remaining UX inconsistencies — no
> redesign, no backend/API/report-generation/PDF/WhatsApp changes.
>
> Property card hierarchy: the card id moved to the top-right as a small gray
> monospace low-emphasis reference (still linking to details) and ₹/sq.ft
> moved under the property name as secondary text, so cards read Price →
> Name → ₹/sq.ft → Location → Specs.
>
> Report History & persistence: every successfully generated report (Reports
> page property reports AND /compare comparison reports) is stored locally
> (features/reports/report-history.ts, localStorage, bounded to 30 — the same
> client-only pattern as conversations.ts). The Reports page is now a report
> center: a Recent Reports list (newest first) with icon, title, date, time,
> property count and an AI Generated badge, plus Preview / Download PDF /
> Share WhatsApp / Delete per report. Preview reopens the stored report text
> instantly; Download renders it through the existing POST /report/pdf; Share
> reuses the existing share endpoints with the stored text — nothing is ever
> regenerated. Reports survive refresh, navigation and returning later.
>
> Profile cleanup: the placeholder Generated Reports cards and the AI Chat
> History section were removed. Profile now links to /reports via a View
> Report History button, and its stats read the same local stores the Reports
> and Chat History pages use.
>
> Chat History page: a dedicated /history page (sidebar entry between AI
> Analysis and Property Comparison) lists the stored Copilot conversations
> (pinned + recent, newest first) with message/property counts; opening one
> resumes it in the Copilot workspace, delete reuses the provider action. UI
> relocation only — no conversation logic changes.
>
> Global search: the navbar placeholder search was replaced with a real input
> wired to a lightweight SearchProvider, shown ONLY on routes with searchable
> content (currently /reports, where it filters the report history live). No
> fake search boxes remain.
>
> Saved Properties: cards render in a new compact PropertyCard density
> (shorter 16/9 image, tighter padding, 3-per-row on desktop) — no
> information removed.
>
> Property Details: feature/nearby/field chips now wrap correctly with a
> maximum chip width and graceful long-text wrapping (no container overflow),
> and section spacing rhythm was made slightly more generous and consistent.
>
> Validation: TypeScript passes, ESLint passes clean, production build
> generates all routes including the new /history.

- [x] Property card hierarchy (id top-right low-emphasis, ₹/sq.ft under name)
- [x] Report History store (localStorage, property + comparison reports)
- [x] Reports page report center (Recent Reports: Preview / Download / Share / Delete, no regeneration)
- [x] Report persistence across refresh/navigation
- [x] Profile cleanup (placeholders removed, View Report History → /reports)
- [x] Dedicated Chat History page at /history + sidebar entry
- [x] Route-scoped functional global search (reports filtering; hidden elsewhere)
- [x] Compact Saved Properties cards (2–3 per row desktop)
- [x] Property Details chips wrap + section spacing polish
- [x] TypeScript · ESLint · production build all pass

---

## Phase 18.10 — Bug Fix: "Show More Properties" Follow-up Flow

> Root cause: the frontend persists conversations in localStorage, but the
> backend's conversational session memory (last_search_filters) is in-process
> and volatile (server restart / TTL). A follow-up like "Show me more such
> properties" on a conversation whose backend session was gone found no
> filters to restore, fell into the DeepSeek generic-chat fallback
> (chat_service.py STEP 5) and returned a huge fabricated markdown answer
> (type "text") instead of search_results. Verified live: same request with a
> live session → search_results (next page); with a dead session → text.
>
> Smallest fix (no search/business-logic/prompt changes): every chat message
> now echoes the last search response's own backend-produced
> `current_query_state` as an optional `last_query_state` request field, and
> the thin API layer re-seeds `session_state["last_search_filters"]` /
> `["last_search_weights"]` from it ONLY when the session lacks them — the
> exact keys the existing follow-up logic already reads, mirroring Streamlit's
> persistent st.session_state. Live sessions, new chats and all other response
> types are untouched (regression-verified over /chat and /chat/stream).

- [x] Root cause traced end-to-end (chip → sendMessage → /chat/stream → parse_intent_and_execute)
- [x] src/api/chat_api.py: optional `last_query_state` + guarded session re-hydration (API layer only)
- [x] types/dashboard.ts: `current_query_state` on search_results, `last_query_state` on ChatRequest
- [x] workspace-provider: echo last search context with each message
- [x] Verified: dead-session follow-up → search_results (real next page, no fabrication)
- [x] Regression: search / comparison / live-session follow-up / new-chat fallback unchanged
- [x] TypeScript · ESLint · production build pass

---

## Phase 18.11 — Reports Layout Refinement & Saved Properties Density Polish

> Presentation-only phase: no backend, API, report-generation, PDF, WhatsApp,
> routing, history-storage or evaluation-tray logic changed — UI moved and
> resized only.

- [x] Share Report moved into the Report Preview toolbar beside Export PDF
      (toggles the existing ShareReportForm as a collapsible panel; the
      always-visible share section below the report removed)
- [x] Reports page two-column layout: report area (~72%) left, sidebar (~28%)
      right with Recent Reports stacked above the Evaluation Tray (same
      ReportHistoryList component; global-search filtering unchanged)
- [x] Compact Saved Properties cards tightened another ~20–25% (shorter 2:1
      image, smaller padding/gaps/badges/price/name/metadata) — no
      information removed, hover animation kept, 3 cards per row desktop
- [x] TypeScript · ESLint · production build pass

---

## Future Enhancements

> These enhancements are outside the current project scope and can be
> implemented after the core application is complete.

- [x] Export reports as PDF (Phase 15.17 preview + Phase 16.2 single Chromium PDF pipeline)
- [ ] Export reports as DOCX
- [ ] Authentication with JWT
- [ ] Persistent database for saved properties
- [ ] Persistent chat/session storage
- [ ] Multi-model AI support (Claude, GPT, Gemini)
- [ ] Voice-based property assistant
- [ ] Image-based property analysis