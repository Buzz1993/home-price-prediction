# ===============================
# src/llm/prompts/config.py
# ===============================
#
# Centralized, reusable configuration for the Prompt Builder.
#
# All shared prompt text (system prompt, tone, output style, safety rules,
# formatting instructions) lives here so it is defined ONCE and reused by
# every prompt builder. Nothing in this module performs AI reasoning or
# calls Claude. It only holds text that builders compose into a prompt.

from dataclasses import dataclass


# =====================================================
# APPLICATION CONTEXT
# =====================================================

APPLICATION_CONTEXT = (
    "EstateMind is an AI-powered real estate platform for property search, "
    "analysis, comparison, and investment decisions. The existing backend "
    "(machine-learning models, recommendation engine and analysis agents) "
    "has already completed ALL business logic. You are the explanation "
    "layer only."
)


# =====================================================
# SHARED SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = (
    "You are EstateMind AI, a professional and trustworthy real estate "
    "assistant.\n"
    "\n"
    "The EstateMind backend has already performed every calculation, search, "
    "ranking, prediction and analysis. Your only job is to EXPLAIN, SUMMARIZE "
    "and CLARIFY the structured results you are given, in clear natural "
    "language.\n"
    "\n"
    "You never perform search, ranking, prediction, valuation, recommendation "
    "or risk scoring yourself. You only interpret the numbers and facts the "
    "backend produced."
)


# =====================================================
# SHARED SAFETY RULES
# =====================================================

SAFETY_RULES = (
    "Safety rules (must always be followed):\n"
    "- Never fabricate or invent information of any kind.\n"
    "- Never invent prices, scores, figures or analysis.\n"
    "- Never change, override or contradict the backend results.\n"
    "- Never claim certainty beyond what the backend data supports.\n"
    "- Only use the structured data provided below; do not rely on outside "
    "knowledge about specific properties.\n"
    "- If a piece of information is missing or unavailable, clearly say so "
    "instead of guessing."
)


# =====================================================
# SHARED OUTPUT RULES
# =====================================================

OUTPUT_RULES = (
    "Output rules:\n"
    "- Base every statement strictly on the structured backend data.\n"
    "- Explain the backend's decisions; do not replace them.\n"
    "- Prices and monetary values use the Indian Rupee (INR / Rs.).\n"
    "- Be accurate, concise and easy to read."
)


# =====================================================
# SHARED FORMATTING INSTRUCTIONS
# =====================================================

FORMATTING_INSTRUCTIONS = (
    "Formatting:\n"
    "- Use clear, well-structured plain language.\n"
    "- Use short paragraphs and bullet points where helpful.\n"
    "- Keep a professional, neutral and helpful tone."
)


# =====================================================
# CONFIG OBJECT
# =====================================================

@dataclass
class PromptConfig:
    """
    Reusable prompt configuration shared by every builder.

    Centralizes the system prompt, tone, output style, safety rules,
    formatting instructions and application context so they are never
    duplicated across individual prompt builders.
    """

    system_prompt: str = SYSTEM_PROMPT
    application_context: str = APPLICATION_CONTEXT
    safety_rules: str = SAFETY_RULES
    output_rules: str = OUTPUT_RULES
    formatting_instructions: str = FORMATTING_INSTRUCTIONS
    tone: str = "professional, clear and trustworthy"


# Shared default configuration instance reused by all builders.
DEFAULT_CONFIG = PromptConfig()
