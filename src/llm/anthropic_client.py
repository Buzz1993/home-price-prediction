# ===============================
# src/llm/anthropic_client.py
# ===============================
#
# Anthropic Claude API transport (Phase 24).
#
# Direct counterpart of src/llm/deepseek_client.py with the SAME contract:
# ask_anthropic / ask_anthropic_stream NEVER raise — failures are returned
# (or yielded) as sentinel error STRINGS so every existing caller handles
# Claude exactly like Ollama:
#
#   "CLAUDE ERROR (<error_type>)\n<message>"          — API-side failure
#   "CLAUDE STREAM ERROR (<error_type>)\n<message>"   — streaming failure
#   "REQUEST EXCEPTION: <message>"                    — unexpected local failure
#   "STREAM EXCEPTION: <message>"                     — unexpected stream failure
#
# <error_type> is a machine-readable category matching ClaudeResponse:
# 'missing_api_key', 'authentication_error', 'rate_limit', 'timeout',
# 'connection_error', 'api_error'.
#
# Configuration (environment variables):
#   ANTHROPIC_API_KEY     — required. Missing key returns a structured
#                           'missing_api_key' sentinel; the backend never crashes.
#   ANTHROPIC_MODEL       — model name (default: claude-sonnet-4-6)
#   ANTHROPIC_MAX_TOKENS  — max output tokens (default: 1024)
#   ANTHROPIC_TEMPERATURE — sampling temperature (default: 0.3)
#   ANTHROPIC_TIMEOUT     — request timeout in seconds (default: 60)

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


ANTHROPIC_MAX_TOKENS = _int_env("ANTHROPIC_MAX_TOKENS", 1024)
ANTHROPIC_TEMPERATURE = _float_env("ANTHROPIC_TEMPERATURE", 0.3)
ANTHROPIC_TIMEOUT = _float_env("ANTHROPIC_TIMEOUT", 60)

MISSING_KEY_SENTINEL = (
    "CLAUDE ERROR (missing_api_key)\n"
    "Claude API key is not configured. Set ANTHROPIC_API_KEY."
)


def _get_client():
    """
    Build an Anthropic client, or None when the API key is missing.
    Imported lazily so the Ollama-only local setup never needs the SDK loaded.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=ANTHROPIC_TIMEOUT)


def _classify_exception(exc: Exception) -> str:
    """Map Anthropic SDK exceptions to the shared error_type categories."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication_error"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit"
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout"
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection_error"
    return "api_error"


def ask_anthropic(prompt):
    """
    Send prompt to the Claude API and return the full response.

    Same contract as ask_deepseek: returns generated text, or a sentinel
    error string on failure. Never raises.
    """

    client = _get_client()
    if client is None:
        return MISSING_KEY_SENTINEL

    try:
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=ANTHROPIC_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        error_type = _classify_exception(e)
        print("\n========== CLAUDE API ERROR ==========")
        print(type(e))
        print(str(e))
        print("======================================\n")
        return f"CLAUDE ERROR ({error_type})\n{str(e)}"

    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


def ask_anthropic_stream(prompt):
    """
    Send prompt to the Claude API and stream the response in chunks.

    Same contract as ask_deepseek_stream: yields text chunks, or a single
    sentinel error string on failure. Never raises.
    """

    client = _get_client()
    if client is None:
        yield MISSING_KEY_SENTINEL
        return

    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=ANTHROPIC_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                if chunk:
                    yield chunk

    except Exception as e:
        error_type = _classify_exception(e)
        print("\n========== CLAUDE STREAM ERROR ==========")
        print(type(e))
        print(str(e))
        print("=========================================\n")
        yield f"CLAUDE STREAM ERROR ({error_type})\n{str(e)}"
