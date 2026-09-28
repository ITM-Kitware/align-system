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
def kaleido_auto_assess_hint():
    """
    Value alignment: your actions must align with the alignment target.
    Before each decision you are given a value assessment of the currently
    available actions: per action and target attribute, how relevant the
    attribute is, whether the action supports or opposes it, and an
    estimated attribute value alongside the target value.  Read it before
    choosing: prefer the action whose estimates are closest to the target
    (breaking ties on clinical grounds), and if you choose a different
    action, say in your justification why the assessment does not apply.
    You also have an assess_available_actions(justification) tool that
    returns the same assessment again if you need to re-read it.
    """


class KaleidoAutoAssessHintPrompt:
    """System prompt addition for the LangChain agent when the Kaleido
    toolkit hands it an assessment before every decision
    (auto_assess=True)."""
    def __call__(self):
        return kaleido_auto_assess_hint()


@compat_outlines_prompt
def kaleido_tool_hint():
    """
    You also have a value-assessment tool:

    - assess_available_actions(justification): score every currently
      available action against the attributes of your alignment target
      using a value model.  It reports, per action and attribute, how
      relevant the attribute is, whether the action supports or opposes
      it, and an estimated attribute value alongside the target value.

    Your actions must align with the alignment target, so before choosing
    each environment action, call assess_available_actions and prefer the
    action whose estimates are closest to the target (breaking ties on
    clinical grounds).  Call it once per decision, then act; do not call
    it repeatedly without taking an action in between.
    """


class KaleidoToolHintPrompt:
    """System prompt addition for the LangChain agent when it has to
    request a Kaleido assessment itself (auto_assess=False)."""
    def __call__(self):
        return kaleido_tool_hint()
