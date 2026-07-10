# ===============================
# src/llm/claude_client.py
# ===============================
#
# Reusable Anthropic Claude service.
#
# Claude is an EXPLANATION layer only. This client knows nothing about
# property search, recommendation, prediction, reports, MCP or FastAPI.
# It only turns a prompt into a text response and returns a structured
# result so callers can handle failures without the backend crashing.
#
# All Claude communication in the project should go through this service.

import os
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables (ANTHROPIC_API_KEY, model config, ...).
load_dotenv()

logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION (environment driven, with safe defaults)
# =====================================================

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.3"))
DEFAULT_TIMEOUT = float(os.getenv("ANTHROPIC_TIMEOUT", "60"))


# =====================================================
# STRUCTURED RESPONSE
# =====================================================

@dataclass
class ClaudeResponse:
    """
    Structured result returned by every Claude call.

    success    : Whether Claude returned usable text.
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
    One event yielded by a streaming Claude call (Phase 15.9).

    Streaming is a DELIVERY enhancement only — it changes how Claude's text
    reaches the caller, not what Claude produces. The event mirrors the same
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
# CLAUDE CLIENT
# =====================================================

class ClaudeClient:
    """
    Thin, reusable wrapper around the Anthropic Claude API.

    The underlying SDK client is created lazily and reused across calls.
    Model, temperature and max tokens are configurable per instance and
    can also be overridden per request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.temperature = (
            DEFAULT_TEMPERATURE if temperature is None else temperature
        )
        self.max_tokens = (
            DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        )
        self.timeout = DEFAULT_TIMEOUT if timeout is None else timeout

        # Lazily initialized single SDK client instance.
        self._client = None

    @property
    def is_configured(self) -> bool:
        """True when an API key is available."""
        return bool(self._api_key)

    def _get_client(self):
        """
        Create (once) and return the shared Anthropic SDK client.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.timeout,
            )
        return self._client

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
        Send a prompt to Claude and return a structured response.

        Args:
            prompt      : User prompt text.
            system      : Optional system instruction.
            model       : Optional model override.
            temperature : Optional temperature override.
            max_tokens  : Optional max tokens override.

        Returns:
            ClaudeResponse: Structured success/error result. This method
            never raises; API failures are returned as structured errors.
        """

        # -- Missing API key ---------------------------------------
        if not self.is_configured:
            logger.error(
                "Claude request failed: ANTHROPIC_API_KEY is not configured."
            )
            return ClaudeResponse(
                success=False,
                error="ANTHROPIC_API_KEY is not configured.",
                error_type="missing_api_key",
            )

        # -- Missing SDK dependency --------------------------------
        try:
            import anthropic
        except ImportError:
            logger.error(
                "Claude request failed: 'anthropic' package is not installed."
            )
            return ClaudeResponse(
                success=False,
                error="The 'anthropic' package is not installed.",
                error_type="missing_dependency",
            )

        resolved_model = model or self.model

        # Never log the prompt content or the API key.
        logger.info("Claude request start (model=%s)", resolved_model)

        try:
            client = self._get_client()

            request_kwargs = {
                "model": resolved_model,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                request_kwargs["system"] = system

            message = client.messages.create(**request_kwargs)

            text = "".join(
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text"
            ).strip()

            # -- Empty response ------------------------------------
            if not text:
                logger.warning("Claude request returned an empty response.")
                return ClaudeResponse(
                    success=False,
                    error="Claude returned an empty response.",
                    error_type="empty_response",
                )

            logger.info("Claude request success (model=%s)", resolved_model)
            return ClaudeResponse(success=True, text=text)

        # -- Invalid / missing credentials -------------------------
        except anthropic.AuthenticationError as exc:
            logger.error("Claude authentication failed: %s", exc)
            return ClaudeResponse(
                success=False,
                error="Invalid Anthropic API key.",
                error_type="authentication_error",
            )

        # -- Rate limiting -----------------------------------------
        except anthropic.RateLimitError as exc:
            logger.error("Claude rate limit exceeded: %s", exc)
            return ClaudeResponse(
                success=False,
                error="Claude rate limit exceeded. Please try again later.",
                error_type="rate_limit",
            )

        # -- Timeout -----------------------------------------------
        except anthropic.APITimeoutError as exc:
            logger.error("Claude request timed out: %s", exc)
            return ClaudeResponse(
                success=False,
                error="Claude request timed out.",
                error_type="timeout",
            )

        # -- Network / connection ----------------------------------
        except anthropic.APIConnectionError as exc:
            logger.error("Claude connection error: %s", exc)
            return ClaudeResponse(
                success=False,
                error="Could not connect to the Claude API.",
                error_type="connection_error",
            )

        # -- Any other API status error ----------------------------
        except anthropic.APIStatusError as exc:
            logger.error("Claude API error (status=%s): %s", exc.status_code, exc)
            return ClaudeResponse(
                success=False,
                error=f"Claude API error (status {exc.status_code}).",
                error_type="api_error",
            )

        # -- Unexpected failure ------------------------------------
        except Exception as exc:  # noqa: BLE001 - never crash the backend
            logger.exception("Unexpected Claude request failure.")
            return ClaudeResponse(
                success=False,
                error=f"Unexpected Claude error: {exc}",
                error_type="unknown_error",
            )

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[ClaudeStreamEvent]:
        """
        Stream a prompt to Claude, yielding text as it is generated (Phase 15.9).

        This is the streaming counterpart of `generate()`. It reuses the SAME
        SDK client, configuration and error categories — only the delivery
        changes: incremental `delta` events arrive as Claude produces them,
        followed by a single terminal `done` (success) or `error` (failure)
        event. The full accumulated text is always available on the terminal
        event, so callers that don't care about streaming can ignore the deltas.

        Args mirror `generate()`. This generator never raises: API failures are
        yielded as an `error` event carrying any partial text collected so far,
        so an interrupted stream can still be shown to the user.
        """

        # -- Missing API key ---------------------------------------
        if not self.is_configured:
            logger.error(
                "Claude stream failed: ANTHROPIC_API_KEY is not configured."
            )
            yield ClaudeStreamEvent(
                type="error",
                error="ANTHROPIC_API_KEY is not configured.",
                error_type="missing_api_key",
            )
            return

        # -- Missing SDK dependency --------------------------------
        try:
            import anthropic
        except ImportError:
            logger.error(
                "Claude stream failed: 'anthropic' package is not installed."
            )
            yield ClaudeStreamEvent(
                type="error",
                error="The 'anthropic' package is not installed.",
                error_type="missing_dependency",
            )
            return

        resolved_model = model or self.model

        # Never log the prompt content or the API key.
        logger.info("Claude stream start (model=%s)", resolved_model)

        # Collect text so the terminal event (and any interrupted stream) can
        # return the partial response.
        collected: list[str] = []

        try:
            client = self._get_client()

            request_kwargs = {
                "model": resolved_model,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                request_kwargs["system"] = system

            # The SDK context manager keeps the upstream connection open only
            # while we iterate. If the caller stops consuming (client
            # disconnect / cancellation), the generator is closed and this
            # `with` block exits, cleanly terminating the Claude request.
            with client.messages.stream(**request_kwargs) as stream:
                for chunk in stream.text_stream:
                    if not chunk:
                        continue
                    collected.append(chunk)
                    yield ClaudeStreamEvent(type="delta", text=chunk)

            text = "".join(collected).strip()

            # -- Empty response ------------------------------------
            if not text:
                logger.warning("Claude stream returned an empty response.")
                yield ClaudeStreamEvent(
                    type="error",
                    error="Claude returned an empty response.",
                    error_type="empty_response",
                )
                return

            logger.info("Claude stream success (model=%s)", resolved_model)
            yield ClaudeStreamEvent(type="done", text=text)

        # -- Invalid / missing credentials -------------------------
        except anthropic.AuthenticationError as exc:
            logger.error("Claude authentication failed: %s", exc)
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error="Invalid Anthropic API key.",
                error_type="authentication_error",
            )

        # -- Rate limiting -----------------------------------------
        except anthropic.RateLimitError as exc:
            logger.error("Claude rate limit exceeded: %s", exc)
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error="Claude rate limit exceeded. Please try again later.",
                error_type="rate_limit",
            )

        # -- Timeout -----------------------------------------------
        except anthropic.APITimeoutError as exc:
            logger.error("Claude stream timed out: %s", exc)
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error="Claude request timed out.",
                error_type="timeout",
            )

        # -- Network / connection ----------------------------------
        except anthropic.APIConnectionError as exc:
            logger.error("Claude connection error: %s", exc)
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error="Could not connect to the Claude API.",
                error_type="connection_error",
            )

        # -- Any other API status error ----------------------------
        except anthropic.APIStatusError as exc:
            logger.error("Claude API error (status=%s): %s", exc.status_code, exc)
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error=f"Claude API error (status {exc.status_code}).",
                error_type="api_error",
            )

        # -- Unexpected failure ------------------------------------
        except Exception as exc:  # noqa: BLE001 - never crash the backend
            logger.exception("Unexpected Claude stream failure.")
            yield ClaudeStreamEvent(
                type="error",
                text="".join(collected),
                error=f"Unexpected Claude error: {exc}",
                error_type="unknown_error",
            )


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
