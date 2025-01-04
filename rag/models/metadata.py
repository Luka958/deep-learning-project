from pydantic import BaseModel
from uuid import UUID


class Metadata(BaseModel):
    id: int | UUID
    text: str