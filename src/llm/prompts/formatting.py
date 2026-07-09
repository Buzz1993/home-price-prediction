# ===============================
# src/llm/prompts/formatting.py
# ===============================
#
# Reusable helpers that turn structured backend data (dicts and lists of
# dicts, exactly as returned by the existing backend services) into clean,
# readable text blocks for a prompt.
#
# These helpers ONLY format data. They never fetch, compute, rank, modify
# or invent anything. Missing values are rendered as "N/A" so the prompt
# can honestly tell Claude when information is unavailable.

from typing import Any


PLACEHOLDER = "N/A"


def format_value(value: Any) -> str:
    """
    Render a single value as readable text.

    None / NaN / empty values become the "N/A" placeholder so Claude is
    never handed a fabricated value.
    """
    if value is None:
        return PLACEHOLDER

    # Detect float NaN without importing pandas/numpy.
    if isinstance(value, float) and value != value:
        return PLACEHOLDER

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return PLACEHOLDER

    return text


def format_record(record: dict, indent: str = "") -> str:
    """
    Render one record (dict) as an aligned "key: value" block.

    Keys are shown in a readable form (snake_case -> Title Case). The order
    of the backend keys is preserved; no field is dropped or renamed.
    """
    if not record:
        return f"{indent}{PLACEHOLDER}"

    lines = []
    for key, value in record.items():
        label = _humanize_key(key)
        lines.append(f"{indent}- {label}: {format_value(value)}")

    return "\n".join(lines)


def format_records(
    records: list[dict],
    item_label: str = "Item",
) -> str:
    """
    Render a list of records (list of dicts) as an enumerated block.

    Used for search results and per-property analysis lists which the
    backend returns as `list[dict]`.
    """
    if not records:
        return PLACEHOLDER

    blocks = []
    for index, record in enumerate(records, start=1):
        if isinstance(record, dict):
            body = format_record(record, indent="  ")
        else:
            body = f"  {format_value(record)}"
        blocks.append(f"{item_label} {index}:\n{body}")

    return "\n\n".join(blocks)


def format_mapping(data: dict) -> str:
    """
    Render a flat mapping (e.g. filters or weights) as an inline list.

    Example:
        bhk: 3bhk | location: Thane | amenities: gym, pool
    """
    if not data:
        return PLACEHOLDER

    parts = [
        f"{_humanize_key(key)}: {format_value(value)}"
        for key, value in data.items()
    ]
    return " | ".join(parts)


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _humanize_key(key: Any) -> str:
    """Turn a snake_case backend key into a readable Title Case label."""
    text = str(key).replace("_", " ").strip()
    return text.title() if text else PLACEHOLDER
