from typing import List

from pydantic import BaseModel


class DialogElement(BaseModel):
    role: str
    content: str


Dialog = List[DialogElement]
