# ===============================
# src/llm/provider_router.py
# ===============================
#
# LLM provider router (Phase 24).
#
# Single entry point for ALL raw LLM transport calls. Callers use
# ask_llm / ask_llm_stream and never know which provider is active:
#
#   LLM_PROVIDER=ollama  -> src/llm/deepseek_client.py  (local default)
#   LLM_PROVIDER=claude  -> src/llm/anthropic_client.py (production/Railway)
#
# Both transports share the same contract: they NEVER raise, and report
# failures as sentinel error STRINGS (see is_llm_error_string below).
# Unknown LLM_PROVIDER values fall back to Ollama so local development
# keeps working exactly as before.

import os

from dotenv import load_dotenv

from src.llm.anthropic_client import ask_anthropic, ask_anthropic_stream
from src.llm.deepseek_client import ask_deepseek, ask_deepseek_stream

load_dotenv()

# Every sentinel prefix either transport can return. Callers that need to
# detect transport failures should use is_llm_error_string() instead of
# hardcoding provider-specific prefixes.
LLM_ERROR_PREFIXES = (
    "REQUEST EXCEPTION",
    "STREAM EXCEPTION",
    "OLLAMA ERROR",
    "OLLAMA STREAM ERROR",
    "CLAUDE ERROR",
    "CLAUDE STREAM ERROR",
)


def get_llm_provider() -> str:
    """Return the active provider: 'claude' or 'ollama' (default)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "claude":
        return "claude"
    return "ollama"


def is_llm_error_string(text) -> bool:
    """True when `text` is a transport sentinel error string."""
    return bool(text) and text.startswith(LLM_ERROR_PREFIXES)


def ask_llm(prompt):
    """
    Send prompt to the configured LLM provider and return the full response.

    Same contract as ask_deepseek / ask_anthropic: returns generated text,
    or a sentinel error string on failure. Never raises.
    """
    if get_llm_provider() == "claude":
        return ask_anthropic(prompt)
    return ask_deepseek(prompt)


def ask_llm_stream(prompt):
    """
    Stream a prompt through the configured LLM provider.

    Same contract as ask_deepseek_stream / ask_anthropic_stream: yields text
    chunks, or a single sentinel error string on failure. Never raises.
    """
    if get_llm_provider() == "claude":
        yield from ask_anthropic_stream(prompt)
    else:
        yield from ask_deepseek_stream(prompt)
