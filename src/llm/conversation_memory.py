# ===============================
# src/llm/conversation_memory.py
# ===============================
#
# Phase 15.7 — Conversational Memory.
#
# A lightweight, SESSION-SCOPED, in-memory conversation store. It lets Claude
# understand follow-up questions within a single active user session ("show the
# cheaper ones", "compare it with the previous one", "explain that again").
#
# What this module IS:
#   - A small per-session record of recent chat turns, the last search context
#     (filters / preferences) and the evaluation-tray property ids.
#   - The mutable `session_state` dict the EXISTING backend workflow
#     (chat_service.parse_intent_and_execute) already reads and writes for its
#     own follow-up handling (last_search_filters, last_search_weights,
#     search_page). We simply keep that dict alive between requests so the
#     existing multi-turn logic works over the stateless HTTP API.
#
# What this module is NOT:
#   - It is NOT a database and NOT persistent. Everything lives in process
#     memory and disappears when the process stops or the session is cleared.
#   - It performs NO business logic: no search, ranking, prediction, valuation
#     or recommendation. It only remembers context so Claude can resolve
#     references. The backend remains the single source of truth.
#
# Memory is OPTIONAL. Callers that cannot obtain a session simply use a fresh
# transient memory and everything degrades to the previous stateless behaviour.

import time
import threading
from dataclasses import dataclass, field

from src.llm.prompts.formatting import format_mapping

# -----------------------------------------------------------------
# Bounds — keep memory lightweight and safe (no unbounded growth).
# -----------------------------------------------------------------
MAX_TURNS = 8                       # recent user/assistant turns kept per session
MAX_SESSIONS = 500                  # hard cap on concurrent sessions
SESSION_TTL_SECONDS = 60 * 60 * 2   # idle sessions expire after 2 hours


# =====================================================
# PER-SESSION MEMORY
# =====================================================

@dataclass
class ConversationMemory:
    """
    Conversation context for a single active session.

    `session_state` is handed straight to the existing backend workflow as its
    `session_state` argument, so the backend's own follow-up/pagination logic
    (last_search_filters, last_search_weights, search_page, ...) keeps working
    across HTTP requests without any change to that logic.
    """

    session_id: str = ""
    session_state: dict = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)      # {"role": ..., "text": ...}
    staged_property_ids: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    # -------------------------------------------------
    # Recording
    # -------------------------------------------------

    def record_user(self, text: str) -> None:
        """Remember the user's latest message."""
        self._append_turn("user", text)

    def record_assistant(self, response: dict, staged_property_ids: list[str] | None = None) -> None:
        """
        Remember a short, non-sensitive summary of the assistant's reply plus
        the evaluation-tray ids that were active for this turn.

        Only a compact label is stored (never full property payloads or backend
        internals), keeping memory lightweight and free of sensitive data.
        """
        self._append_turn("assistant", _summarize_response(response))
        if staged_property_ids is not None:
            # Keep the most recent view of the tray only.
            self.staged_property_ids = list(staged_property_ids)
        self.last_updated = time.time()

    def _append_turn(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.turns.append({"role": role, "text": text})
        # Retain only the most recent turns.
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]
        self.last_updated = time.time()

    # -------------------------------------------------
    # Prompt context
    # -------------------------------------------------

    def summary(self) -> str | None:
        """
        Build a compact, human-readable memory block for the prompt builder.

        Returns None when there is nothing worth adding, so the caller can send
        a smaller prompt. Only relevant context is included — never the entire
        conversation payload.
        """
        sections: list[str] = []

        # Recent turns (oldest first, most recent last) — excluding the current
        # user message which the caller passes separately as the user intent.
        prior_turns = self.turns[:-1] if self.turns else []
        if prior_turns:
            lines = [
                f"- {'User' if t['role'] == 'user' else 'Assistant'}: {t['text']}"
                for t in prior_turns[-MAX_TURNS:]
            ]
            sections.append("Recent conversation (oldest first):\n" + "\n".join(lines))

        # Last search context the backend recorded for follow-up searches.
        filters = self.session_state.get("last_search_filters")
        if filters:
            sections.append("Last search filters: " + format_mapping(filters))

        preferences = self.session_state.get("last_search_preferences")
        if preferences:
            sections.append("Stated preferences: " + format_mapping(preferences))

        # Properties the user is actively evaluating / comparing.
        if self.staged_property_ids:
            sections.append(
                "Properties in the evaluation tray: "
                + ", ".join(str(pid) for pid in self.staged_property_ids)
            )

        if not sections:
            return None

        return "\n".join(sections)


# =====================================================
# RESPONSE SUMMARY (compact, non-sensitive)
# =====================================================

def _summarize_response(response: dict) -> str:
    """Turn a structured backend response into a short memory label."""
    if not isinstance(response, dict):
        return ""

    rtype = response.get("type")
    content = response.get("content")

    if rtype == "text":
        text = str(content or "").strip()
        return text[:280]

    if rtype == "search_results":
        count = len(content) if isinstance(content, list) else 0
        return f"Returned {count} ranked propert{'y' if count == 1 else 'ies'} for the search."

    # Tray-based analyses (comparison, rental, prediction, negotiation,
    # valuation, advisor) — label only; the numbers stay in the backend result.
    if rtype:
        return f"Provided a {rtype} result for the staged properties."

    return ""


# =====================================================
# SESSION STORE (in-memory, session-scoped)
# =====================================================

class SessionMemoryStore:
    """
    Thread-safe, in-memory registry of per-session ConversationMemory objects.

    Session-scoped only: entries are evicted on TTL expiry or when the session
    cap is exceeded, and can be cleared explicitly. There is no persistence.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> ConversationMemory:
        """
        Return the memory for `session_id`, creating it on first use.

        When `session_id` is falsy, a fresh transient (unstored) memory is
        returned so the caller behaves exactly like the previous stateless flow.
        """
        if not session_id:
            return ConversationMemory()

        with self._lock:
            self._prune_expired()
            memory = self._sessions.get(session_id)
            if memory is None:
                self._enforce_capacity()
                memory = ConversationMemory(session_id=session_id)
                self._sessions[session_id] = memory
            return memory

    def clear(self, session_id: str | None) -> None:
        """Forget a single session (e.g. on sign-out or a new chat)."""
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    # -------------------------------------------------
    # Internal maintenance
    # -------------------------------------------------

    def _prune_expired(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        expired = [
            sid for sid, mem in self._sessions.items()
            if mem.last_updated < cutoff
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def _enforce_capacity(self) -> None:
        # Evict the least-recently-updated sessions until under the cap.
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(
                self._sessions.items(),
                key=lambda item: item[1].last_updated,
            )[0]
            self._sessions.pop(oldest, None)


# =====================================================
# SHARED SINGLETON ACCESSOR
# =====================================================

_default_store: SessionMemoryStore | None = None


def get_session_memory_store() -> SessionMemoryStore:
    """Return the shared, lazily-created in-memory session store."""
    global _default_store
    if _default_store is None:
        _default_store = SessionMemoryStore()
    return _default_store
