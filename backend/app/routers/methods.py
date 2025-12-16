from __future__ import annotations

from typing import List

from fastapi import APIRouter

from ..core.loaders import list_methods
from ..schemas.method import Method

router = APIRouter(prefix="/methods", tags=["methods"])


@router.get("/", response_model=List[Method])
def read_methods() -> List[Method]:
    return [Method(**method) for method in list_methods()]
