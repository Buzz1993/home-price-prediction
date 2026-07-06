# System Architecture

Version: 1.0

---

# 1. Overall Architecture

EstateMind uses a layered architecture where the Next.js frontend communicates with the EstateMind Copilot FastAPI API.

The EstateMind API is a thin layer that exposes the existing backend services without duplicating business logic.

The backend contains all AI, Machine Learning, search, recommendation, and analysis logic.

The frontend is responsible only for the user interface and consuming backend APIs.

```text
┌──────────────────────────────┐
│     Next.js Frontend         │
│    React + TypeScript        │
└───────────────┬──────────────┘
                │
           REST APIs
                │
┌───────────────▼──────────────┐
│   EstateMind Copilot API     │
│      src/api/main.py         │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│   API Router                 │
│ src/api/copilot_api.py       │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│      property_tools.py       │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────┐
│ mcp_real_estate_service.py   │
└───────────────┬──────────────┘
                │
 ┌──────────────┼──────────────────────────────────────────────────────────────┐
 │              │              │              │             │                  │
 ▼              ▼              ▼              ▼             ▼                  ▼
Search      Recommendation  Analysis     Prediction    Reports &          MCP Tools
Engine         Engine         Agents       Service       Sharing
                                   │
                                   ▼
                          Machine Learning Models
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

The frontend communicates with the EstateMind Copilot API.

The EstateMind API exposes the existing backend functionality through thin FastAPI endpoints.

Business logic remains in the existing backend services.

---

# 4. EstateMind API Layer

The frontend communicates with the EstateMind Copilot API.

Location:

- src/api/main.py
- src/api/copilot_api.py

The API layer is intentionally thin.

Responsibilities:

- Validate requests
- Route requests
- Call existing backend services
- Return responses

The API layer must not contain business logic.

Business logic remains in:

- property_tools.py
- mcp_real_estate_service.py
- Search Engine
- Recommendation Engine
- Analysis Agents
- Prediction Service

---

# 5. Current Application

The existing Streamlit application is the reference implementation.

All user workflows, business logic, and backend integrations already exist.

The Next.js application should reproduce the same functionality with a modern UI.

Do not redesign workflows unless required.

---

# 6. Authentication Flow

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

# 7. AI Chat Pipeline

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

# 8. Property Search Pipeline

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

# 9. Hybrid Recommendation Pipeline

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

# 10. ML Prediction Pipeline

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

# 11. Analysis Agents

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

# 12. Report Generation Pipeline

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

# 13. Report Sharing Pipeline

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

# 14. Data Flow

```text
User
      │
      ▼
Next.js Frontend
      │
      ▼
EstateMind Copilot API
(src/api/main.py)
      │
      ▼
property_tools.py
      │
      ▼
mcp_real_estate_service.py
      │
      ▼
Backend Services
(Search, Recommendation,
Analysis, Prediction,
Reports)
      │
      ▼
Response
      │
      ▼
Frontend
```

---

# 15. Project Structure

```text
EstateMind/
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── copilot_api.py
│   │
│   ├── agents/
│   ├── core/
│   ├── data/
│   ├── graph/
│   ├── llm/
│   ├── mcp/
│   ├── models/
│   ├── recommender/
│   ├── services/
│   ├── streamlit_app/
│   ├── ui/
│   ├── utils/
│   └── visualization/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── features/
│   ├── hooks/
│   ├── lib/
│   ├── services/
│   ├── types/
│   └── utils/
│
├── project_docs/
│   ├── 01_Project.md
│   ├── 02_System.md
│   ├── 03_API.md
│   ├── 04_UI.md
│   ├── 05_Features.md
│
├── app.py                  # ML Prediction API
├── CLAUDE.md
└── README.md
```

---

# 16. Frontend Modules

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