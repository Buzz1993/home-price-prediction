# EstateMind Copilot

Version: 1.0

---

# 1. Project Overview

EstateMind Copilot is an AI-powered real estate platform.

Users can search, compare, and analyze residential properties using Artificial Intelligence and Machine Learning.

The Python backend is already implemented and considered the source of truth for this project.

The frontend must integrate with the existing backend instead of recreating backend logic.

This project focuses on building a modern Next.js web application around the existing backend.

---

# Existing Backend Platform

The Python backend is already implemented and should be reused.

The backend already provides the following capabilities:

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
* Report Sharing (Phone Number)
* MCP Tools
* n8n Integration

The backend also includes:

* Search Registry
* BM25 Search
* Hybrid Ranking
* Multi-Agent Analysis Pipeline
* Prediction Services
* Report Services

These modules are already implemented and should be reused.

Do not recreate or redesign them.

The frontend should only consume the existing backend APIs.

---

# Current Application

The existing Streamlit application is the reference implementation.

All user workflows, business logic, and backend integrations already exist.

The Next.js frontend should reproduce the same functionality with a modern, responsive UI.

Reuse the existing workflows and backend APIs.

Do not redesign workflows or business logic unless required.

---

# Scope

This project focuses only on building the Next.js frontend.

The existing Python backend, ML models, AI agents, search engine, recommendation engine, and report generation pipeline must not be modified.

---

# Existing Search Pipeline

Property search is already implemented in the backend.

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

The frontend should display the search results returned by the backend.

Do not implement search or ranking logic in the frontend.

---

# Existing Analysis Agents

The backend already contains specialized analysis agents for:

* Price Analysis
* Rental Analysis
* Risk Analysis
* Future Growth Analysis
* Property Valuation
* Property Comparison
* Investment Advisor
* Negotiation Assistant

The frontend should display these results and never implement the analysis itself.

---

# 2. Problem Statement

Most real estate websites only display property listings.

EstateMind helps users make informed buying decisions using:

* AI Search
* Price Prediction
* Property Valuation
* Rental Analysis
* Investment Insights
* Property Comparison

---

# 3. Goals

Build a modern web application that allows users to:

* Search properties
* Chat with AI
* Compare properties
* View AI-powered insights
* Generate reports
* Share reports to a phone number
* Save favourite properties
* Manage user accounts

---

# 4. Target Users

* Home Buyers
* Property Investors
* First-Time Buyers
* Property Consultants

---

# 5. Tech Stack

## Frontend

* Next.js
* TypeScript
* Tailwind CSS
* shadcn/ui
* TanStack Query

## Backend

Existing Python Backend

* FastAPI
* Machine Learning Models
* AI Services
* Recommendation Engine
* Multi-Agent System
* MCP Tools

The frontend must reuse the existing backend.

Never rewrite backend business logic.

---

# 6. Core Features

* User Authentication
* AI Chat
* AI Property Search
* Hybrid Recommendation
* Property Details
* Property Comparison
* Price Prediction
* Rental Analysis
* Property Valuation
* Risk Analysis
* Future Growth Analysis
* Investment Advisor
* Negotiation Assistant
* AI Report Generation
* Report Sharing (Phone Number)
* Dashboard
* Saved Properties
* User Profile

---

# 7. User Journey

```text
Landing Page

↓

Login / Sign Up

↓

Dashboard

↓

Search Properties

↓

View Property Details

↓

Compare Properties

↓

AI Analysis

↓

Generate Report

↓

Share Report to Phone Number
```

---

# 8. Folder Structure

```text
EstateMind/

backend/
    Existing Python Backend

frontend/
    app/
    components/
    features/
    hooks/
    lib/
    services/
    types/

docs/
    01_Project.md
    02_System.md
    03_API.md
    04_UI.md
    05_Features.md
    CLAUDE.md

README.md
```