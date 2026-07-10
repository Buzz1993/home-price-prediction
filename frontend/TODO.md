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
- [ ] Explain Future Growth — BLOCKED: no backend future-growth endpoint or tool exists; Claude never invents analysis.
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