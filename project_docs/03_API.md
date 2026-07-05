# API Documentation

Version: 1.0

---

# Overview

The frontend communicates with the existing Python backend through REST APIs.

The existing Python backend is the source of truth.

It already provides:

- Intent Extraction
- AI Chat
- Hybrid Property Search
- Recommendation Engine
- Analysis Agents
- Price Prediction
- Report Generation
- MCP Tools
- n8n Integration

Reuse the existing backend whenever possible.

Do not rewrite backend business logic.

If an API is unavailable, ask for clarification instead of creating a new implementation.

---

# API Rules

These endpoints describe the expected frontend integration.

If an endpoint is not yet exposed by the backend:

- Reuse the existing backend functionality.
- Expose a thin FastAPI endpoint if needed.
- Do not duplicate backend business logic.

---

# Authentication APIs

## POST /signup

**Purpose:**
Create a new user account.

**Input:**

- Name
- Email
- Password

**Output:**

- User Account

---

## POST /login

**Purpose:**
Authenticate user.

**Input:**

- Email
- Password

**Output:**

- Authentication Token

---

## POST /logout

**Purpose:**
Logout user.

**Input:**

- Authentication Token

**Output:**

- Success Status

---

## GET /profile

**Purpose:**
Retrieve user profile.

**Output:**

- User Profile

---

# AI Chat API

## POST /chat

**Purpose:**
Process natural language requests and route them to the appropriate backend service.

**Input:**

- User Message

**Output:**

- AI Response

---

# Property Search API

## POST /search

**Purpose:**
Search properties.

**Input:**

- City
- Location
- Budget
- BHK

**Output:**

- Property List

---

# Property Details API

## GET /property/{id}

**Purpose:**
Retrieve complete property details.

**Input:**

- Property ID

**Output:**

- Property Details

---

# Comparison API

## POST /compare

**Purpose:**
Compare selected properties.

**Input:**

- Property IDs

**Output:**

- Comparison Result

---

# Price Prediction API

## POST /predict

**Purpose:**
Predict property price.

**Input:**

- Property ID

**Output:**

- Prediction Result

---

# Rental Analysis API

## POST /rental

**Purpose:**
Analyze rental potential.

**Input:**

- Property ID

**Output:**

- Rental Analysis

---

# Valuation API

## POST /valuation

**Purpose:**
Analyze property valuation.

**Input:**

- Property ID

**Output:**

- Valuation Result

---

# Investment Advisor API

## POST /advisor

**Purpose:**
Generate investment advice.

**Input:**

- Property ID

**Output:**

- Investment Recommendation

---

# Negotiation API

## POST /negotiation

**Purpose:**
Generate negotiation strategy.

**Input:**

- Property ID

**Output:**

- Negotiation Advice

---

# Report API

## POST /report

**Purpose:**
Generate an AI property report.

**Input:**

- Property IDs

**Output:**

- Report Result

---

# Share Report API

## POST /report/share

**Purpose:**
Send a generated report to a phone number.

**Input:**

- Phone Number
- Report

**Output:**

- Delivery Status

---

# Saved Properties APIs

## GET /saved-properties

**Purpose:**
Retrieve saved properties.

**Output:**

- Saved Property List

---

## POST /save-property

**Purpose:**
Save a property.

**Input:**

- Property ID

**Output:**

- Success Status

---

## DELETE /save-property

**Purpose:**
Remove a saved property.

**Input:**

- Property ID

**Output:**

- Success Status

---

# Chat History API

## GET /chat-history

**Purpose:**
Retrieve previous AI conversations.

**Output:**

- Chat History

---

# Reports API

## GET /reports

**Purpose:**
Retrieve previously generated reports.

**Output:**

- Report List

---

# API Design Rules

- Reuse the existing backend APIs.
- Do not rewrite backend business logic.
- Do not invent backend business logic.
- Keep API contracts unchanged.
- Use TanStack Query for all API requests.
- Handle loading, success, empty, and error states.
- Ask for clarification if an API or backend dependency is unavailable.
- Keep frontend service functions thin and delegate all business logic to the backend.

---

# Backend Integration Rules

The backend already contains the required business logic.

If an API endpoint is missing:

1. Reuse the existing backend service.
2. Expose it through a thin FastAPI endpoint.
3. Do not duplicate business logic.
4. Keep the frontend independent of backend implementation details.