# Features Documentation

Version: 1.0

---

# Overview

EstateMind provides AI-powered tools to help users search, analyze, compare, and evaluate real estate properties.

The existing Python backend already implements all feature logic.

The frontend should consume the existing APIs and display the results.

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
```

**Input**

- Search Filters

**Output**

- Property List

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

# 7. Price Prediction

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

# 8. Rental Analysis

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

# 9. Property Valuation

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

# 10. Risk Analysis

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

# 11. Future Growth Analysis

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

# 12. Investment Advisor

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

# 13. Negotiation Assistant

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

# 14. Report Generation

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

# 15. Report Sharing

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

# 16. Saved Properties

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

# 17. Chat History

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