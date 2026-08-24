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


@compat_outlines_prompt
def followup_clarify_treatment(character, supplies):
    """
    Given the description below, please state what treatment supply \
    should be applied to the patient given their injuries and vitals. \
    Select a treatment from the following list of available supplies \
    (quantities shown in parentheses):
    {% for supply, qty in supplies.items() %}
    - {{ supply }} (x{{ qty }})
    {% endfor %}

    Description:
    {{ character.unstructured }}
    """

class FollowupClarifyTreatmentPrompt:
    def __call__(self, character, supplies):
        return followup_clarify_treatment(character, supplies)


@compat_outlines_prompt
def ow_part3_character_description_w_vitals(c):
    """
    {{ c.name }}: {{ c.unstructured }}
    {% if c.vitals is none or (c.vitals.avpu is none and c.vitals.breathing is none and c.vitals.heart_rate is none) %}
      Vitals: Unknown
    {% else %}
      Vitals:
      {% if c.vitals.avpu is not none %}
        - AVPU: {{ c.vitals.avpu.value if c.vitals.avpu.value is defined else c.vitals.avpu }}
      {% endif %}
      {% if c.vitals.breathing is not none %}
        - Breathing: {{ c.vitals.breathing.value if c.vitals.breathing.value is defined else c.vitals.breathing }}
      {% endif %}
      {% if c.vitals.heart_rate is not none %}
        - Heart Rate: {{ c.vitals.heart_rate.value if c.vitals.heart_rate.value is defined else c.vitals.heart_rate }}
      {% endif %}
    {% endif %}
    """

class OWPart3CharacterDescriptionWVitals:
    def __call__(self, scenario_state):
        return ow_part3_character_description_w_vitals(scenario_state)
