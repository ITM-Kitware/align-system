from typing import List

from pydantic import BaseModel


class DialogElement(BaseModel):
    role: str
    content: str

    # def __getitem__(self, item):
    #     if isinstance(item, str):
    #         return getattr(self,item)
        

Dialog = List[DialogElement]
