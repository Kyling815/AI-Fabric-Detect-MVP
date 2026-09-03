from fastapi import APIRouter

from backend.app.db.schemas import FabricListItem, FabricListResponse
from backend.app.services.fabric_registry import FabricRegistry


router = APIRouter()


@router.get("/fabrics", response_model=FabricListResponse)
def list_fabrics() -> FabricListResponse:
    registry = FabricRegistry()
    items = []
    for record in registry.list_fabrics():
        status = str(record.get("status", "INACTIVE")).upper()
        active_reference = record.get("active_reference")
        items.append(
            FabricListItem(
                fabric_id=str(record["fabric_id"]),
                display_name=str(record.get("display_name") or record["fabric_id"]),
                status=status,
                active_reference=str(active_reference) if active_reference else None,
                inspectable=status == "ACTIVE" and bool(active_reference),
            )
        )
    return FabricListResponse(fabrics=items)
