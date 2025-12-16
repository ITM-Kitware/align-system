from typing import Any, Union, Optional

from pydantic import BaseModel


class AttributeValidValueRange(BaseModel):
    min: int
    max: int
    step: int


class AttributeValidValues(BaseModel):
    values: list[int]


class Attribute(BaseModel):
    name: str
    kdma: str
    description: Optional[str] = None

    factor: Optional[int] = None
    score_examples: Optional[str] = None
    valid_scores: Optional[Union[AttributeValidValueRange, AttributeValidValues]] = None

    relevant_structured_character_info: Optional[list[str]] = None


class AttributeTarget(Attribute):
    value: Optional[float]=None
    parameters: Optional[list[dict[str, Any]]]=None
