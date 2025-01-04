from pydantic import BaseModel


class Metadata(BaseModel):
    id: str
    text: str