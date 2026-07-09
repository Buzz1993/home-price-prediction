# API Documentation

Version: 1.0

---

# Overview

The frontend communicates with the EstateMind Copilot FastAPI API.

The EstateMind API is a thin layer that exposes the existing Python backend as REST endpoints.

The existing Python backend remains the source of truth for all business logic.

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

The EstateMind API layer is implemented in:

- src/api/main.py
- src/api/analysis_api.py

These files expose the existing backend services without duplicating business logic.

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

Ranked Property List

Each property may include:

- property_id
- image_urls
- ap_pjt_url
- latitude
- longitude
- project_name
- locality
- city
- bed
- bath
- parking
- balcony
- area
- cost_per_sqft
- price
- hybrid_score

The frontend should render all available search results on the interactive map using the returned latitude and longitude coordinates.

---

# Property Details API

## GET /property/{id}

**Purpose:**
Retrieve complete property details.

**Input:**

- Property ID

**Output:**

Property Details

Includes all property metadata available in the backend.

The frontend should render:

- All image_urls
- ap_pjt_url
- Amenities
- Features
- Nearby Places
- Reviews
- Ratings
- Coordinates
- Pricing
- Builder
- Project
- Unknown fields

Examples include:

- property_id
- image_urls
- ap_pjt_url
- project_name
- locality
- city
- bed
- bath
- parking
- balcony
- area
- cost_per_sqft
- price
- hybrid_score
- property_type
- furnish
- status
- amenities
- Additional property metadata returned by the backend.

The frontend should automatically categorize backend fields into logical sections instead of relying on a fixed schema.

The frontend should display every field returned by the backend whenever possible.

The frontend should dynamically render all backend fields without requiring API changes when additional property metadata becomes available.

Unknown fields should not be ignored. Instead, they should be grouped into logical sections such as:

- Property Overview
- Pricing
- Property Specifications
- Project Information
- Amenities
- Features
- Nearby Places
- Reviews
- Location
- AI Insights
- Additional Information

---

# Analysis Request

All analysis endpoints accept the following request body:

```json
{
  "property_ids": [
    "cardid69427147"
  ]
}
```

**Note**

- Most analysis endpoints accept one property ID.
- `POST /analysis/comparison` requires at least two property IDs.

---

# Comparison API

## POST /analysis/comparison

**Purpose:**
Compare selected properties.

**Input:**

- property_ids (minimum two property IDs)

**Output:**

- Comparison Result

---

# Price Prediction API

## POST /analysis/predict

**Purpose:**
Predict property price.

**Input:**

- property_ids (list containing one property ID)

**Output:**

- Prediction Result

---

# Rental Analysis API

## POST /analysis/rental

**Purpose:**
Analyze rental potential.

**Input:**

- property_ids (list containing one property ID)

**Output:**

- Rental Analysis

---

# Valuation API

## POST /analysis/valuation

**Purpose:**
Analyze property valuation.

**Input:**

- property_ids (list containing one property ID)

**Output:**

- Valuation Result

---

# Investment Advisor API

## POST /analysis/advisor

**Purpose:**
Generate investment advice.

**Input:**

- property_ids (list containing one property ID)

**Output:**

- Investment Recommendation

---

# Negotiation API

## POST /analysis/negotiation

**Purpose:**
Generate negotiation strategy.

**Input:**

- property_ids (list containing one property ID)

**Output:**

- Negotiation Advice

---

# Report API

## POST /report

**Purpose:**
Generate an AI property report.

**Input:**

- property_ids (list of property IDs)

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

## POST /analysis/cache/clear

**Purpose:**
Clear the backend enrichment cache.

**Input:**
None

**Output:**
Success Status

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
- The frontend should gracefully render optional fields when they are available and hide them when they are not.

---

# Backend Integration Rules

The backend already contains the required business logic.

If an API endpoint is missing:

1. Reuse the existing backend service.
2. Expose it through a thin FastAPI endpoint.
3. Do not duplicate business logic.
4. Keep the frontend independent of backend implementation details.