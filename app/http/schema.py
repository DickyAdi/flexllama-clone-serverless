from pydantic import BaseModel
from typing import List


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    documents: List[str]