from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Method(BaseModel):
    id: str
    name: str
    category: str
    latency: str
    strengths: List[str]
    limitations: List[str]
