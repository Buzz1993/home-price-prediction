# ===============================
# src/llm/claude_client.py
# ===============================
#
# Reusable LLM service (EXPLANATION layer only).
#
# The public interface (ClaudeResponse, ClaudeStreamEvent, ClaudeClient,
# get_claude_client, ask_claude, stream_claude) is preserved so existing
# callers keep working unchanged. Internally all transport goes through the
# provider router (src/llm/provider_router.py: ask_llm / ask_llm_stream),
# which selects Ollama or the Claude API based on LLM_PROVIDER (Phase 24).
#
# This service knows nothing about property search, recommendation,
# prediction, reports, MCP or FastAPI. It only turns a prompt into a text
# response and returns a structured result so callers can handle failures
# without the backend crashing. All LLM communication should go through here.

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from src.llm.provider_router import ask_llm, ask_llm_stream, get_llm_provider

logger = logging.getLogger(__name__)


# =====================================================
# STRUCTURED RESPONSE
# =====================================================

@dataclass
class ClaudeResponse:
    """
    Structured result returned by every LLM call.

    success    : Whether the model returned usable text.
    text       : The generated natural-language response.
    error      : Human-readable error message (when success is False).
    error_type : Machine-readable error category, e.g.
                 'missing_api_key', 'authentication_error',
                 'rate_limit', 'timeout', 'connection_error',
                 'api_error', 'empty_response'.
    """

    success: bool
    text: str = ""
    error: str | None = None
    error_type: str | None = None


@dataclass
class ClaudeStreamEvent:
    """
    One event yielded by a streaming LLM call (Phase 15.9).

    Streaming is a DELIVERY enhancement only — it changes how the text
    reaches the caller, not what the model produces. The event mirrors the same
    error categories as ClaudeResponse so callers handle failures identically.

    type       : 'delta' (an incremental text chunk),
                 'done'  (stream finished; `text` holds the full response),
                 'error' (the stream failed; `error`/`error_type` are set and
                          `text` holds any partial text accumulated so far).
    text       : Incremental chunk (delta) or full/partial text (done/error).
    error      : Human-readable error message (when type is 'error').
    error_type : Machine-readable error category, matching ClaudeResponse.
    """

    type: str
    text: str = ""
    error: str | None = None
    error_type: str | None = None


# =====================================================
# TRANSPORT RESULT ADAPTER
# =====================================================
#
# ask_llm / ask_llm_stream never raise: they return (or yield) an error
# STRING with a known prefix instead. These helpers map those strings to
# the structured error categories used by ClaudeResponse / ClaudeStreamEvent.

def _compose_prompt(prompt: str, system: str | None) -> str:
    """
    Merge an optional system instruction and the user prompt into a single
    prompt string, since the transport clients accept one prompt only.
    """
    if system:
        return f"{system}\n\n{prompt}"
    return prompt


def _classify_transport_error(text: str) -> str | None:
    """
    Return the error_type for a transport sentinel error string, or None
    when the text is a normal (successful) response.
    """
    if text.startswith("REQUEST EXCEPTION:") or text.startswith("STREAM EXCEPTION:"):
        return "connection_error"
    if text.startswith("OLLAMA ERROR") or text.startswith("OLLAMA STREAM ERROR"):
        return "api_error"
    if text.startswith("CLAUDE ERROR") or text.startswith("CLAUDE STREAM ERROR"):
        # The Claude transport embeds the category in the sentinel itself,
        # e.g. "CLAUDE ERROR (missing_api_key)\n...".
        match = re.match(r"CLAUDE (?:STREAM )?ERROR \(([a-z_]+)\)", text)
        return match.group(1) if match else "api_error"
    return None


# =====================================================
# LLM CLIENT
# =====================================================

class ClaudeClient:
    """
    Thin, reusable wrapper around the LLM backend.

    Backed by the provider router (Ollama locally, Claude API in production).
    The constructor keeps its previous signature for backward compatibility,
    but model / temperature / max_tokens / timeout are managed by the active
    transport (via environment variables) and are accepted here only so
    existing construction and call sites keep working.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        # Retained for interface compatibility; not used by the DeepSeek backend.
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """
        Whether the active provider can serve requests. Ollama needs no API
        key; a missing ANTHROPIC_API_KEY on the claude provider surfaces as a
        structured 'missing_api_key' error from generate()/stream() rather
        than a crash, so this stays permissive.
        """
        return True

    # -------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ClaudeResponse:
        """
        Send a prompt to the LLM and return a structured response.

        Args:
            prompt      : User prompt text.
            system      : Optional system instruction (prepended to the prompt).
            model       : Accepted for compatibility; ignored by the backend.
            temperature : Accepted for compatibility; ignored by the backend.
            max_tokens  : Accepted for compatibility; ignored by the backend.

        Returns:
            ClaudeResponse: Structured success/error result. This method
            never raises; failures are returned as structured errors.
        """

        full_prompt = _compose_prompt(prompt, system)

        provider = get_llm_provider()
        logger.info("LLM request start (provider=%s)", provider)

        try:
            raw = ask_llm(full_prompt)
        except Exception as exc:  # noqa: BLE001 - never crash the backend
            logger.exception("Unexpected LLM request failure.")
            return ClaudeResponse(
                success=False,
                error=f"Unexpected LLM error: {exc}",
                error_type="unknown_error",
            )

        text = (raw or "").strip()

        # -- Sentinel error string returned by the transport ------------
        error_type = _classify_transport_error(text)
        if error_type:
            logger.error(
                "LLM request failed (provider=%s, error_type=%s).", provider, error_type
            )
            return ClaudeResponse(
                success=False,
                error=text,
                error_type=error_type,
            )

        # -- Empty response --------------------------------------------
        if not text:
            logger.warning("LLM request returned an empty response.")
            return ClaudeResponse(
                success=False,
                error="The LLM returned an empty response.",
                error_type="empty_response",
            )

        logger.info("LLM request success (provider=%s)", provider)
        return ClaudeResponse(success=True, text=text)

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[ClaudeStreamEvent]:
        """
        Stream a prompt to the LLM, yielding text as it is generated (Phase 15.9).

        Streaming counterpart of `generate()`, backed by the provider
        router's `ask_llm_stream`. Incremental `delta` events arrive as the
        model produces them, followed by a single terminal `done` (success) or
        `error` (failure) event. The full accumulated text is always available
        on the terminal event.

        This generator never raises: failures are yielded as an `error` event
        carrying any partial text collected so far.
        """

        full_prompt = _compose_prompt(prompt, system)

        provider = get_llm_provider()
        logger.info("LLM stream start (provider=%s)", provider)

        # Collect text so the terminal event (and any interrupted stream) can
        # return the partial response.
        collected: list[str] = []

        try:
            for chunk in ask_llm_stream(full_prompt):
                if not chunk:
                    continue

                # The stream yields a single sentinel error string on failure.
                error_type = _classify_transport_error(chunk)
                if error_type:
                    logger.error(
                        "LLM stream failed (provider=%s, error_type=%s).",
                        provider,
                        error_type,
                    )
                    yield ClaudeStreamEvent(
                        type="error",
                        text="".join(collected),
                        error=chunk,
                        error_type=error_type,
                    )
                    return

                collected.append(chunk)
                yield ClaudeStreamEvent(type="delta", text=chunk)

        except Exception as exc:  # noqa: BLE001 - never crash the backend
            logger.exception("Unexpected LLM stream failure.")
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error=f"Unexpected LLM error: {exc}",
                error_type="unknown_error",
            )
            return

        text = "".join(collected).strip()

        # -- Empty response --------------------------------------------
        if not text:
            logger.warning("LLM stream returned an empty response.")
            yield ClaudeStreamEvent(
                type="error",
                error="The LLM returned an empty response.",
                error_type="empty_response",
            )
            return

        logger.info("LLM stream success (provider=%s)", provider)
        yield ClaudeStreamEvent(type="done", text=text)


# =====================================================
# MARKDOWN SANITIZER (UI explanations)
# =====================================================

def strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting from LLM output so UI-card explanations render
    as plain text. Unwraps **bold**/*italic* markers and strips heading
    hashes, keeping the text content itself unchanged. Currency symbols,
    '•' bullets and normal punctuation are untouched. NOT applied to the
    report enhancer (Markdown by design) or to JSON tool outputs.
    """
    text = re.sub(r"^\s*```[\w-]*\s*$", "", text or "", flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    return text.replace("*", "").replace("#", "").replace("`", "").strip()


# =====================================================
# SHARED SINGLETON ACCESSORS
# =====================================================

_default_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    """
    Return a shared, lazily-created ClaudeClient instance.
    """
    global _default_client
    if _default_client is None:
        _default_client = ClaudeClient()
    return _default_client


def ask_claude(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ClaudeResponse:
    """
    Convenience helper that sends a prompt through the shared client.
    """
    return get_claude_client().generate(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def stream_claude(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[ClaudeStreamEvent]:
    """
    Streaming counterpart of `ask_claude` — yields ClaudeStreamEvent items
    through the shared client (Phase 15.9).
    """
    return get_claude_client().stream(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
