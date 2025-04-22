from typing import List

from pydantic import BaseModel


class DialogElement(BaseModel):
    role: str
    content: str

    tags: List[str]


Dialog = List[DialogElement]
