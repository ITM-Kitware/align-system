import jinja2
import json
import re

from align_system.prompt_engineering.compat.outlines.templates import prompt as compat_outlines_prompt


@compat_outlines_prompt
def followup_clarify_tag(character):
    """
    Given the casualty description below, please state what triage tag \
    should be applied to the casualty given their injuries and vitals

    {{ character.unstructured }}
    """

class FollowupClarifyTagPrompt:
    def __call__(self, character):
        return followup_clarify_tag(character)


@compat_outlines_prompt
def followup_clarify_character(scenario_description, action):
    """
    Given the scenario description and selected action, please clarify which \
    character should receive the action.

    SCENARIO DESCRIPTION:
    {{ scenario_descriptoin }}

    SELECTED ACTION:
    {{ action.unstructured }}
    """

class FollowupClarifyCharacterPrompt:
    def __call__(self, scenario_description, action):
        return followup_clarify_character(scenario_description, action)
