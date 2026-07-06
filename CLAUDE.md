# CLAUDE.md

# Project

EstateMind is an AI-powered real estate platform for property search, analysis, comparison, and investment decisions.

The Python backend is already complete.

Your task is to build only the Next.js frontend.

---

# Project Goal

This project is a portfolio project for an entry-level Machine Learning Engineer / Data Scientist.

Priorities:

1. Clean architecture
2. Readable code
3. Reusable components
4. Simple implementation
5. Maintainable code

Avoid enterprise-level complexity unless necessary.

---

# Development Environment

## Python Backend

The existing Python backend uses the virtual environment `.venv2`.

Always use:

```text
.venv2\Scripts\python.exe
```

Examples:

```text
.venv2\Scripts\python.exe -m pip install -r requirements.txt

.venv2\Scripts\python.exe -m streamlit run streamlit_app/main.py
```

---

## Frontend

The frontend is located in the `frontend/` directory.

Use Node.js and npm (or pnpm if I explicitly request it) for all frontend development.

Examples:

```text
cd frontend

npm install

npm run dev

npm run build
```

Do not use the Python environment for frontend development.

---

# Communication Style

- Keep responses concise.
- Build only the requested task.
- Do not explain well-known concepts unless asked.
- Do not generate unnecessary code.
- Reuse existing code whenever possible.
- Stop after completing the requested task.
- Ask for clarification if requirements are ambiguous.

---

# Session Workflow

For every new feature:

1. Start a new chat.
2. Read CLAUDE.md.
3. Read only the relevant documentation.
4. Review the corresponding Streamlit implementation.
5. Build only one feature.
6. Test the feature.
7. Update `frontend/TODO.md` if the completed task changes its status.
8. Stop after the requested task.

---

# Existing Backend Platform

The Python backend is already implemented.

Before implementing or testing frontend features, start the required backend services.

### Machine Learning Prediction API

```text
.venv2\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

### EstateMind Copilot API

Start the EstateMind Copilot API:

```text
.venv2\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8001
```

Configure the frontend (`frontend/.env.local`):

```text
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
```

The Next.js frontend communicates with the EstateMind Copilot API on port **8001**.

The EstateMind Copilot API is a thin orchestration layer.

It exposes the existing backend functionality as REST APIs.

When required, it delegates price prediction to the Machine Learning Prediction API (port **8000**).

Do not implement business logic in the EstateMind Copilot API.

Only expose existing backend functionality through thin FastAPI endpoints.

---

# Existing Streamlit Application

The existing Streamlit application is the reference implementation and the source of truth.

All user workflows, business logic, and backend integrations already exist.

The Next.js application should reproduce the same functionality with a modern UI.

When implementing a page, reproduce the existing Streamlit functionality before introducing UI improvements.

Do not redesign workflows unless required.

---

# Tech Stack

## Frontend

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- React Hook Form
- Zod
- TanStack Query

## Backend

- FastAPI
- Python
- Machine Learning Models
- Multi-Agent System
- MCP Tools
- n8n Webhooks

---

# Folder Structure

Follow the project structure defined in `02_System.md`.

---

# Build Order

Build one feature at a time.

Follow the order defined in frontend/TODO.md.

---

# Development Rules

- Build one page at a time.
- Do not modify existing backend business logic.
- The EstateMind API layer (src/api) may be extended only to expose existing backend functionality. Do not implement new business logic in this layer.
- Consume backend APIs only.
- Keep components reusable.
- Keep code simple and readable.
- Avoid unnecessary libraries.
- Use TypeScript everywhere.
- Do not over-engineer the frontend.
- Prefer simple React patterns over complex abstractions.
- A straightforward implementation is preferred over maximum flexibility.

---

# Architecture Rules

Frontend is independent.

```text
Next.js
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
Existing Backend Services
      │
      ▼
Prediction API (app.py)
      │
      ▼
Machine Learning Models
```

Never move business logic to the frontend.

---

# UI Rules

Use:

- Tailwind CSS
- shadcn/ui

Design should be:

- Clean
- Modern
- Responsive
- Minimal

Prefer a premium dashboard style with reusable UI components.

Avoid excessive animations.

---

# Component Rules

- Create small, reusable components.
- Prefer composition over duplication.
- Reuse existing components whenever possible.

---

# State Management

Use:

- React Context (Global UI State)
- TanStack Query (API State)
- Local Component State

Avoid Redux.

---

# API Rules

Never hardcode data.

Always fetch data from backend APIs.

Handle:

- Loading
- Success
- Empty State
- Error State

Do not duplicate backend business logic.

---

# Authentication Rules

Protected pages require login.

Store authentication tokens securely.

Redirect unauthenticated users to the Login page.

---

# Scope Control

Implement only what I request.

Do not:

- Build future features.
- Refactor unrelated code.
- Improve files outside the requested task.
- Add optional functionality unless asked.

---

# Do Not Build

Do not create:

- New backend business logic
- New database schemas
- Admin panels
- Notification systems
- Payment systems
- Deployment infrastructure
- CI/CD pipelines
- Microservices

Existing backend services may be exposed through thin FastAPI endpoints when required.

---

# Working Style

When implementing a feature:

- Build only the requested feature.
- Reuse existing backend APIs.
- Reuse existing components whenever possible.
- Do not modify unrelated files.
- Keep components small and reusable.
- Ask before changing project architecture.
- Prefer simple solutions over clever ones.

---

# Important

- The Streamlit application is the source of truth.
- Do not redesign the backend.
- Do not change API contracts.
- Do not recreate existing backend features.
- Do not duplicate backend business logic.
- Do not invent APIs or database schemas.
- Ask for clarification if an API or dependency is missing.
- Focus on clean, maintainable frontend code.
- Do not continue to the next feature unless I explicitly request it.

---

# Before Coding

For every task:

1. Read `CLAUDE.md`.
2. Read only the required documentation.
3. Review the matching Streamlit implementation.
4. If the task requires backend integration:

   Ensure the following services are running:

   - ML Prediction API (port 8000)
   - EstateMind Copilot API (port 8001)

   Ensure the frontend is configured to use:

   ```text
   frontend/.env.local

   NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
   ```

5. Reuse the existing backend APIs.
6. Build only the requested task.
7. Do not modify unrelated files.
8. Stop after completing the requested task.

---

# Source of Truth

This repository contains multiple Machine Learning experiments, utilities, and legacy code.

For the EstateMind Copilot frontend:

1. Start with `src/streamlit_app/pages/estatemind_copilot.py`.
2. Inspect only the files imported by it.
3. Reuse the existing backend implementation. Do not recreate business logic.
4. Do not inspect, modify, or reference unrelated files unless I explicitly instruct you.
5. If required information is missing, stop and ask for clarification before accessing additional files.
6. Ignore all experiments, notebooks, training pipelines, and legacy code unless I explicitly request them.

---

# Out of Scope

Unless I explicitly request otherwise, ignore:

### Streamlit Pages

- analytics.py
- recommendations.py
- price_prediction.py

### Directories

- notebooks/
- models/
- reports/
- references/
- tests/
- deploy/

### Code

- Legacy code
- Experimental code
- Debugging code
- Training pipelines

Only use files defined in the **Source of Truth** section.

If additional backend information is required, stop and ask for clarification before inspecting other files.