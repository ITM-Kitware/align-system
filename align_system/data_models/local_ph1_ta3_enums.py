import json
from enum import Enum
from typing_extensions import Self


# COPY/PASTE data models from Phase 1 ITM Evaluation client code; as
# the scenario data structure changes for new domains and evaluations
# we need to keep a copy of these around for backward compatibility


class InjuryStatusEnum(str, Enum):
    """
    Whether the injury is known prior- and post-assessment, and to what extent it's been treated
    """

    """
    allowed enum values
    """
    HIDDEN = 'hidden'
    DISCOVERABLE = 'discoverable'
    VISIBLE = 'visible'
    DISCOVERED = 'discovered'
    PARTIALLY_TREATED = 'partially treated'
    TREATED = 'treated'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of InjuryStatusEnum from a JSON string"""
        return cls(json.loads(json_str))


class ActionTypeEnum(str, Enum):
    """
    An action type recognized by the ADM Server, combining basic and domain-specific actions
    """

    """
    allowed enum values
    """
    END_SCENE = 'END_SCENE'
    MOVE_TO = 'MOVE_TO'
    MESSAGE = 'MESSAGE'
    SEARCH = 'SEARCH'
    APPLY_TREATMENT = 'APPLY_TREATMENT'
    CHECK_ALL_VITALS = 'CHECK_ALL_VITALS'
    CHECK_BLOOD_OXYGEN = 'CHECK_BLOOD_OXYGEN'
    CHECK_PULSE = 'CHECK_PULSE'
    CHECK_RESPIRATION = 'CHECK_RESPIRATION'
    DIRECT_MOBILE_CHARACTERS = 'DIRECT_MOBILE_CHARACTERS'
    MOVE_TO_EVAC = 'MOVE_TO_EVAC'
    SITREP = 'SITREP'
    TAG_CHARACTER = 'TAG_CHARACTER'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of ActionTypeEnum from a JSON string"""
        return cls(json.loads(json_str))


class InjuryLocationEnum(str, Enum):
    """
    the injury location on the character's body
    """

    """
    allowed enum values
    """
    RIGHT_FOREARM = 'right forearm'
    LEFT_FOREARM = 'left forearm'
    RIGHT_HAND = 'right hand'
    LEFT_HAND = 'left hand'
    RIGHT_LEG = 'right leg'
    LEFT_LEG = 'left leg'
    RIGHT_CALF = 'right calf'
    LEFT_CALF = 'left calf'
    RIGHT_THIGH = 'right thigh'
    LEFT_THIGH = 'left thigh'
    RIGHT_STOMACH = 'right stomach'
    LEFT_STOMACH = 'left stomach'
    RIGHT_BICEP = 'right bicep'
    LEFT_BICEP = 'left bicep'
    RIGHT_SHOULDER = 'right shoulder'
    LEFT_SHOULDER = 'left shoulder'
    RIGHT_SIDE = 'right side'
    LEFT_SIDE = 'left side'
    RIGHT_CHEST = 'right chest'
    LEFT_CHEST = 'left chest'
    CENTER_CHEST = 'center chest'
    RIGHT_WRIST = 'right wrist'
    LEFT_WRIST = 'left wrist'
    LEFT_FACE = 'left face'
    RIGHT_FACE = 'right face'
    LEFT_NECK = 'left neck'
    RIGHT_NECK = 'right neck'
    INTERNAL = 'internal'
    HEAD = 'head'
    NECK = 'neck'
    STOMACH = 'stomach'
    UNSPECIFIED = 'unspecified'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of InjuryLocationEnum from a JSON string"""
        return cls(json.loads(json_str))


class CharacterTagEnum(str, Enum):
    """
    the tag assigned to a character
    """

    """
    allowed enum values
    """
    MINIMAL = 'MINIMAL'
    DELAYED = 'DELAYED'
    IMMEDIATE = 'IMMEDIATE'
    EXPECTANT = 'EXPECTANT'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of CharacterTagEnum from a JSON string"""
        return cls(json.loads(json_str))
