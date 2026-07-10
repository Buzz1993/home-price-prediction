# ===============================
# src/llm/prompts/__init__.py
# ===============================
#
# Prompt Builder package (Phase 15.2).
#
# Converts structured backend results into consistent, high-quality prompts
# for Claude. This package only FORMATS backend data into prompt strings.
# It performs no AI reasoning and never calls Claude or any backend service.
#
# Typical usage (in later phases):
#
#     from src.llm.prompts import build_search_prompt
#     from src.llm.claude_client import ask_claude
#
#     prompt = build_search_prompt(user_query, results, filters, weights)
#     response = ask_claude(prompt.user, system=prompt.system)

from src.llm.prompts.config import (
    PromptConfig,
    DEFAULT_CONFIG,
    SYSTEM_PROMPT,
    APPLICATION_CONTEXT,
    SAFETY_RULES,
    OUTPUT_RULES,
    FORMATTING_INSTRUCTIONS,
)
from src.llm.prompts.templates import (
    Prompt,
    build_prompt,
    compose_user_prompt,
)
from src.llm.prompts.search_prompt import build_search_prompt
from src.llm.prompts.analysis_prompt import build_analysis_prompt
from src.llm.prompts.comparison_prompt import build_comparison_prompt
from src.llm.prompts.investment_prompt import build_investment_prompt
from src.llm.prompts.report_prompt import build_report_prompt
from src.llm.prompts.orchestration_prompt import (
    ToolSpec,
    build_orchestration_prompt,
    ROUTER_SYSTEM_PROMPT,
)
from src.llm.prompts.suggestions_prompt import (
    build_suggestions_prompt,
    SUGGESTIONS_SYSTEM_PROMPT,
)

__all__ = [
    # Configuration
    "PromptConfig",
    "DEFAULT_CONFIG",
    "SYSTEM_PROMPT",
    "APPLICATION_CONTEXT",
    "SAFETY_RULES",
    "OUTPUT_RULES",
    "FORMATTING_INSTRUCTIONS",
    # Shared templates
    "Prompt",
    "build_prompt",
    "compose_user_prompt",
    # Builders
    "build_search_prompt",
    "build_analysis_prompt",
    "build_comparison_prompt",
    "build_investment_prompt",
    "build_report_prompt",
    # Tool orchestration (Phase 15.8)
    "ToolSpec",
    "build_orchestration_prompt",
    "ROUTER_SYSTEM_PROMPT",
    # AI suggestions (Phase 15.11)
    "build_suggestions_prompt",
    "SUGGESTIONS_SYSTEM_PROMPT",
]
