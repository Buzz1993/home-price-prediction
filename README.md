# EstateMind Copilot

AI-powered real estate platform for intelligent property search, comparison, valuation, and investment analysis.

## Features

- Intent Extraction
- Hybrid Property Search
- Recommendation Engine
- Property Comparison
- Property Price Prediction
- Rental Analysis
- Property Valuation
- Risk Analysis
- Future Growth Analysis
- Investment Advisor
- Negotiation Assistant
- AI Report Generation
- Report Sharing
- MCP Tools

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- TanStack Query

### Backend
- FastAPI
- Python
- Machine Learning
- Multi-Agent System
- Ollama
- MCP

## Email Delivery (Password Reset)

Password reset emails use a hybrid transport. The provider is selected
automatically from environment variables — no code changes needed:

- **Local development — Gmail SMTP**: set `SMTP_HOST`, `SMTP_PORT`,
  `SMTP_USERNAME`, `SMTP_PASSWORD` (Gmail App Password), `SMTP_FROM`,
  and `FRONTEND_BASE_URL`.
- **Production (Railway) — Resend HTTPS API**: set `RESEND_API_KEY`,
  `EMAIL_FROM`, and `FRONTEND_BASE_URL`. Railway blocks outbound SMTP
  ports (25/465/587), so email is delivered over HTTPS via Resend.

If `RESEND_API_KEY` is set, Resend is used; otherwise Gmail SMTP.
See `.env.example` for details.

## Project Structure

```text
frontend/
project_docs/
src/
data/
cache/