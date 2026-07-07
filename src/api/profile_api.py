# ===============================
# src/api/profile_api.py
# ===============================

from fastapi import APIRouter

router = APIRouter(
    tags=["Profile"]
)

# =====================================================
# DEMO DATA
# =====================================================

REPORTS = [
    {
        "id": "R001",
        "title": "Investment Report",
        "created_at": "2026-07-01"
    },
    {
        "id": "R002",
        "title": "Property Comparison",
        "created_at": "2026-07-03"
    }
]

CHAT_HISTORY = [
    {
        "role": "user",
        "message": "Show me 2 BHK properties in Thane."
    },
    {
        "role": "assistant",
        "message": "Here are the best matching properties."
    },
    {
        "role": "user",
        "message": "Compare the first two."
    },
    {
        "role": "assistant",
        "message": "Comparison completed successfully."
    }
]



# =====================================================
# REPORT HISTORY
# =====================================================

@router.get("/reports")
def get_reports():
    """
    Return generated reports.
    """

    return REPORTS


# =====================================================
# CHAT HISTORY
# =====================================================

@router.get("/chat-history")
def get_chat_history():
    """
    Return previous chat messages.
    """

    return CHAT_HISTORY