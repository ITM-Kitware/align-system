from typing import Union, List

from pydantic import BaseModel


class AttributeValidValueRange(BaseModel):
    min: int
    max: int
    step: int


class AttributeValidValues(BaseModel):
    values: List[int]


class Attribute(BaseModel):
    name: str
    description: str

    factor: int
    score_examples: str
    valid_scores: Union[AttributeValidValueRange, AttributeValidValues]

    relevant_structured_character_info: List[str]
