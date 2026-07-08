# Features Documentation

Version: 1.0

---

# Overview

EstateMind provides AI-powered tools to help users search, analyze, compare, and evaluate real estate properties.

The existing Python backend already implements all feature logic.

All AI analysis features are exposed through the EstateMind Copilot REST API.

The frontend consumes these APIs and displays the returned results.

The frontend never implements business logic or AI analysis.

Each feature includes:

- Purpose
- Workflow
- Input
- Output

---

# 1. Authentication

**Purpose:**
Allow users to securely access the application.

**Workflow**

```text
Sign Up
    ↓
Login
    ↓
Dashboard
```

**Input**

- Name
- Email
- Password

**Output**

- Authenticated User

---

# 2. AI Chat

**Purpose:**
Search and analyze properties using natural language.

**Workflow**

```text
User Query
      ↓
Intent Extraction
      ↓
Property Search / Analysis Routing
      ↓
Backend Services
      ↓
AI Response / Search Results
```

**Input**

- User Query

**Output**

- AI Response

---

# 3. Property Search

**Purpose:**
Find matching properties.

**Workflow**

```text
Search Query

↓

Hybrid Search

↓

Ranked Results

↓

Interactive Map
```

**Input**

- Search Filters

**Output**

Ranked Property Cards

Each Property Card displays:

- Primary Image
- Property Summary
- Investment Score
- Price
- Original Listing Link

Search results are also displayed on an interactive map.

Default View

- Show all properties returned by the backend.
- Display price-only markers.

Filtered View

- Show only matched properties.
- Display rich markers containing:
  - Primary Image
  - Price
  - Cost per Sqft
  - Area
  - BHK

---

# 4. Evaluation Tray

**Purpose:**
Temporarily store selected properties for AI analysis.

**Workflow**

```text
Property Search
      ↓
Select Property
      ↓
Evaluation Tray
```

---

# 5. Hybrid Recommendation

**Purpose:**
Recommend the best matching properties.

**Workflow**

```text
Search Results
      ↓
Recommendation Engine
      ↓
Recommended Properties
```

**Input**

- Search Results

**Output**

- Recommended Properties

---

# 6. Property Comparison

**Purpose:**
Compare multiple properties.

**Workflow**

```text
Select Properties
      ↓
Comparison
      ↓
Recommendation
```

**Input**

- Property IDs

**Output**

- Comparison Result

---

# 7. Property Details

**Purpose:**
Display complete property information for a selected property.

**Workflow**

```text
Property

↓

Property Details

↓

Property Information

↓

Image Gallery

↓

Original Listing

↓

AI Analysis
```

**Input**

- Property ID

**Output**

- Complete Property Details
- Organized Property Information
- Property Specifications
- Amenities
- Image Gallery
- Original Listing
- AI Analysis

---

# 8. Property Image Gallery

**Purpose:**
Provide an interactive property image experience.

**Workflow**

```text
Property
      ↓
image_urls
      ↓
Gallery
      ↓
Fullscreen Viewer
      ↓
Image Navigation
```

**Input**

- image_urls

**Output**

- Image Gallery
- Fullscreen Viewer
- Image Carousel

---

# 9. Property Information

Purpose

Display all backend property information in organized sections.

Workflow

Property

↓

Property Details

↓

Organized Information

↓

Amenities

↓

Location

↓

AI Analysis

Input

Complete property metadata returned by the backend.

Output

- Property Overview
- Pricing
- Property Specifications
- Project Information
- Amenities
- Location
- Additional Information

The frontend should organize all backend property metadata into appropriate sections automatically.

---

# 10. Property Map

**Purpose:**
Visualize property locations on an interactive map.

**Workflow**

```text
Property Search
      ↓
Property Results
      ↓
Map Markers
      ↓
Property Preview
      ↓
Property Details
```

**Input**

- latitude
- longitude
- image_urls
- price
- cost_per_sqft
- area
- bed

**Output**

- Interactive Map
- Price Markers
- Rich Property Preview

---

# 11. Price Prediction

**Purpose:**
Predict property value.

**Workflow**

```text
Property
      ↓
Prediction Service
      ↓
Prediction Result
```

**Input**

- Property ID

**Output**

- Prediction Result

---

# 12. Rental Analysis

**Purpose:**
Analyze rental potential.

**Workflow**

```text
Property
      ↓
Rental Analysis
      ↓
Result
```

**Input**

- Property ID

**Output**

- Rental Analysis

---

# 13. Property Valuation

**Purpose:**
Evaluate market valuation.

**Workflow**

```text
Property
      ↓
Valuation Analysis
      ↓
Result
```

**Input**

- Property ID

**Output**

- Valuation Result

---

# 14. Risk Analysis

**Purpose:**
Evaluate investment risks.

**Workflow**

```text
Property
      ↓
Risk Analysis
      ↓
Result
```

**Input**

- Property ID

**Output**

- Risk Analysis

---

# 15. Future Growth Analysis

**Purpose:**
Estimate future appreciation.

**Workflow**

```text
Property
      ↓
Growth Analysis
      ↓
Result
```

**Input**

- Property ID

**Output**

- Growth Analysis

---

# 16. Investment Advisor

**Purpose:**
Generate investment recommendations.

**Workflow**

```text
Property
      ↓
Analysis Agents
      ↓
Investment Advisor
      ↓
Recommendation
```

**Input**

- Property ID

**Output**

- Investment Recommendation

---

# 17. Negotiation Assistant

**Purpose:**
Suggest a negotiation strategy.

**Workflow**

```text
Property
      ↓
Negotiation Analysis
      ↓
Recommendation
```

**Input**

- Property ID

**Output**

- Negotiation Advice

---

# 18. Report Generation

**Purpose:**
Generate an AI property report.

**Workflow**

```text
Property
      ↓
Generate Report
      ↓
Report
```

**Input**

- Property IDs

**Output**

- Property Report

---

# 19. Report Sharing

**Purpose:**
Send a generated report to a phone number.

**Workflow**

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
Report Sent Successfully
```

**Input**

- Phone Number
- Report

**Output**

- Delivery Status

---

# 20. Saved Properties

**Purpose:**
Save favourite properties.

**Workflow**

```text
Property
      ↓
Save
      ↓
Saved List
```

**Input**

- Property ID

**Output**

- Saved Property

---

# 21. Chat History

**Purpose:**
View previous AI conversations.

**Workflow**

```text
Conversation
      ↓
History
```

**Input**

- User

**Output**

- Chat History

---

# Overall Feature Flow

```text
Authentication
      ↓
AI Chat
      ↓
Intent Extraction
      ↓
Hybrid Property Search
      ↓
Interactive Property Map
      ↓
Recommendation Engine
      ↓
Evaluation Tray
      ↓
Property Comparison
      ↓
Analysis Agents
      ├── Price Prediction
      ├── Rental Analysis
      ├── Property Valuation
      ├── Risk Analysis
      ├── Future Growth
      ├── Investment Advisor
      └── Negotiation Assistant
      ↓
Report Generation
      ↓
Report Sharing
      ↓
Saved Properties
      ↓
Chat History
```