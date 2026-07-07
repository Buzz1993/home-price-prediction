# ===============================
# src/api/saved_api.py
# ===============================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(
    tags=["Saved Properties"]
)

# =====================================================
# DEMO STORAGE
# =====================================================

saved_properties = []


# =====================================================
# REQUEST MODEL
# =====================================================

class SavePropertyRequest(BaseModel):
    property_id: str


# =====================================================
# GET SAVED PROPERTIES
# =====================================================

@router.get("/saved-properties")
def get_saved_properties():
    """
    Return all saved property IDs.
    """

    return {
        "saved_properties": saved_properties
    }


# =====================================================
# SAVE PROPERTY
# =====================================================

@router.post("/save-property")
def save_property(request: SavePropertyRequest):
    """
    Save a property.
    """

    if request.property_id not in saved_properties:
        saved_properties.append(request.property_id)

    return {
        "success": True,
        "message": "Property saved successfully.",
        "saved_properties": saved_properties
    }


# =====================================================
# REMOVE SAVED PROPERTY
# =====================================================

@router.delete("/save-property")
def remove_saved_property(request: SavePropertyRequest):
    """
    Remove a saved property.
    """

    if request.property_id not in saved_properties:
        raise HTTPException(
            status_code=404,
            detail="Property not found in saved list."
        )

    saved_properties.remove(request.property_id)

    return {
        "success": True,
        "message": "Property removed successfully.",
        "saved_properties": saved_properties
    }