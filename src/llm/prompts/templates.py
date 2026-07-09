# ===============================
# src/llm/prompts/templates.py
# ===============================
#
# Shared prompt building blocks reused by every prompt builder.
#
# This module defines:
#   - The `Prompt` result object (system + user text).
#   - A single `compose_user_prompt` helper that lays out the canonical
#     prompt structure so every builder produces a consistent prompt.
#
# It performs NO AI reasoning and makes NO API calls. It only assembles
# plain-text sections.

from dataclasses import dataclass

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG


# =====================================================
# PROMPT RESULT OBJECT
# =====================================================

@dataclass
class Prompt:
    """
    A built prompt as plain text.

    system : Shared system instruction (role + safety framing).
    user   : The composed user prompt containing context, backend data
             and task instructions.

    Both fields are plain strings. `to_text()` returns the two combined,
    which is useful for clients that accept a single prompt string.
    The Claude client from Phase 15.1 accepts `system` and `prompt`
    separately, so `Prompt.system` and `Prompt.user` map directly onto it.
    """

    system: str
    user: str

    def to_text(self) -> str:
        """Return the system and user prompt combined into one string."""
        return f"{self.system}\n\n{self.user}"

    def __str__(self) -> str:
        return self.to_text()


# =====================================================
# CANONICAL PROMPT STRUCTURE
# =====================================================

def compose_user_prompt(
    user_intent: str,
    backend_data: str,
    task_instructions: str,
    expected_output: str,
    config: PromptConfig = DEFAULT_CONFIG,
) -> str:
    """
    Assemble the user prompt using the canonical EstateMind structure:

        Application Context
        ->
        User Intent
        ->
        Structured Backend Data
        ->
        Instructions for Claude
        ->
        Safety Rules
        ->
        Output & Formatting Rules
        ->
        Expected Output Style

    Every builder uses this helper so all prompts share the same shape.

    Args:
        user_intent       : What the user is trying to do / their question.
        backend_data       : Pre-formatted structured backend results.
        task_instructions  : What Claude should do with the data.
        expected_output    : Description of the expected output style.
        config             : Shared prompt configuration (defaults to
                             DEFAULT_CONFIG).

    Returns:
        str: The composed user prompt as plain text.
    """

    sections = [
        _section("APPLICATION CONTEXT", config.application_context),
        _section("USER INTENT", user_intent),
        _section("STRUCTURED BACKEND DATA", backend_data),
        _section("YOUR TASK", task_instructions),
        _section("SAFETY RULES", config.safety_rules),
        _section("OUTPUT RULES", config.output_rules),
        _section("FORMATTING", config.formatting_instructions),
        _section("EXPECTED OUTPUT STYLE", expected_output),
    ]

    return "\n\n".join(sections).strip()


def build_prompt(
    user_intent: str,
    backend_data: str,
    task_instructions: str,
    expected_output: str,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Convenience wrapper that returns a full `Prompt` (system + user).

    Shared by every builder so the system prompt is attached consistently.
    """

    user = compose_user_prompt(
        user_intent=user_intent,
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )

    return Prompt(system=config.system_prompt, user=user)


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _section(title: str, body: str) -> str:
    """Render a titled section with a consistent header style."""
    body = (body or "").strip()
    return f"## {title}\n{body}"
