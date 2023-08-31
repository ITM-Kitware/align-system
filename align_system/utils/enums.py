from enum import Enum


class ProbeType(Enum):
    MultipleChoice = "MultipleChoice"
    FreeResponse = "FreeResponse"
    PatientOrdering = "PatientOrdering"
    SelectTag = "SelectTag"
    SelectTreatment = "SelectTreatment"
