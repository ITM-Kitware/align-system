import jinja2
import json
import re

from align_system.prompt_engineering.compat.outlines.templates import prompt as compat_outlines_prompt


@compat_outlines_prompt
def followup_clarify_tag(character):
    """
    Given the description below, please state what triage category \
    should be applied to the patient given their injuries and vitals. \
    When selecting a triage category, consider the following criteria:
    - IMMEDIATE: Life-threatening injuries requiring immediate intervention \
    and likely to survive if treated promptly.
    - DELAYED: Serious injuries requiring medical care but treatment can \
    safely be delayed.
    - MINOR: Walking wounded or patients with minor injuries who can safely \
    wait.
    - EXPECTANT: Injuries so severe that survival is unlikely given available \
    resources during the incident. This category should be used sparingly.

    Description:
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

    Scenario:
    {{ scenario_descriptoin }}

    Selected action:
    {{ action.unstructured }}
    """

class FollowupClarifyCharacterPrompt:
    def __call__(self, scenario_description, action):
        return followup_clarify_character(scenario_description, action)
