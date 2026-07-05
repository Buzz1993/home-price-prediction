# System Architecture

Version: 1.0

---

# 1. Overall Architecture

EstateMind uses a layered architecture where the Next.js frontend communicates with the existing Python backend.

The backend already contains all AI, Machine Learning, search, recommendation, and analysis logic.

The frontend is responsible only for the user interface and consuming backend APIs.

```text
┌──────────────────────────┐
│        Frontend          │
│    Next.js + React       │
└─────────────┬────────────┘
              │
          REST APIs
              │
┌─────────────▼────────────┐
│ Existing Python Backend  │
│        FastAPI           │
└─────────────┬────────────┘
              │
 ┌────────────┼────────────────────────────────────────────────────────────┐
 │            │             │              │             │                 │
 ▼            ▼             ▼              ▼             ▼                 ▼
Intent     Search     Recommendation   Analysis      ML Models     Reports &
Engine     Engine        Engine          Agents                     MCP Tools
              │
              ▼
      Property Dataset
```

---

# 2. Frontend Architecture

The frontend is responsible for:

* Authentication
* Dashboard
* AI Chat
* Property Search
* Property Details
* Property Comparison
* Reports
* Saved Properties
* User Profile

The frontend only displays data.

All business logic remains in the backend.

The frontend should not contain any Machine Learning, search, recommendation, or business logic.

Its responsibilities are:

- Display data
- Collect user input
- Call backend APIs
- Manage UI state

---

# 3. Backend Platform

The existing Python backend is already implemented and should be reused.

The backend provides:

* Intent Extraction
* AI Chat
* Hybrid Property Search
* Recommendation Engine
* Property Comparison
* Price Prediction
* Property Valuation
* Rental Analysis
* Risk Analysis
* Future Growth Analysis
* Investment Advisor
* Negotiation Assistant
* Report Generation
* Report Sharing
* MCP Tools
* n8n Integration

These modules are already implemented.

Do not recreate or redesign them.

The frontend should only consume the existing backend APIs.

---

# 4. Current Application

The existing Streamlit application is the reference implementation.

All user workflows, business logic, and backend integrations already exist.

The Next.js application should reproduce the same functionality with a modern UI.

Do not redesign workflows unless required.

---

# 5. Authentication Flow

```text
User

↓

Sign Up / Login

↓

Authentication

↓

Dashboard

↓

Protected Pages
```

Each authenticated user has:

* Profile
* Saved Properties
* Chat History
* Generated Reports

---

# 6. AI Chat Pipeline

```text
User Query

↓

Intent Extraction

↓

Backend Service

↓

LLM Response / Results

↓

Frontend
```

The backend already performs:

* Intent Extraction
* Context Management
* Natural Language Understanding
* Response Generation

The frontend only renders the conversation.

---

# 7. Property Search Pipeline

The search pipeline already exists in the backend.

```text
User Query

↓

Intent Extraction

↓

Property Search

↓

Hybrid Ranking

↓

Recommendation

↓

Results
```

The frontend displays the returned results.

Search logic should never be implemented in the frontend.

---

# 8. Hybrid Recommendation Pipeline

The recommendation engine combines multiple ranking strategies.

```text
Property Search

↓

Hybrid Ranking

↓

Recommendation Engine

↓

Ranked Results
```

Recommendation logic remains in the backend.

---

# 9. ML Prediction Pipeline

```text
Property

↓

Prediction Service

↓

Machine Learning Model

↓

Prediction Result
```

The frontend only displays prediction results.

---

# 10. Analysis Agents

The backend already contains specialized analysis agents for:

* Price Analysis
* Property Comparison
* Rental Analysis
* Property Valuation
* Risk Analysis
* Future Growth Analysis
* Investment Advisor
* Negotiation Strategy

The frontend should display these results and never implement the analysis itself.

---

# 11. Report Generation Pipeline

```text
Selected Properties

↓

Generate Report

↓

Preview Report

↓

Download Report
```

Report generation is handled by the backend.

---

# 12. Report Sharing Pipeline

```text
Compare Properties

↓

Generate AI Report

↓

Preview Report

↓

Enter Phone Number

↓

Send Report

↓

MCP Tool

↓

n8n Workflow

↓

Report Sent Successfully
```

The backend already supports report sharing.

The frontend only collects the phone number and displays the delivery status.

---

# 13. Data Flow

```text
User

↓

Next.js Frontend

↓

REST APIs

↓

Backend Services

↓

Response

↓

Frontend
```

---

# 14. Project Structure

```text
frontend/
│
├── app/
├── components/
├── features/
├── hooks/
├── lib/
├── services/
├── types/
└── utils/

backend/
│
└── Existing Python Backend

docs/
│
├── 01_Project.md
├── 02_System.md
├── 03_API.md
├── 04_UI.md
├── 05_Features.md
└── CLAUDE.md
```

---

# 15. Frontend Modules

* Authentication
* Dashboard
* AI Chat
* Property Search
* Property Details
* Property Comparison
* Reports
* Saved Properties
* Profile

Each module should be independent and reusable.