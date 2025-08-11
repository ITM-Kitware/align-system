from swagger_client.models import (
    ActionTypeEnum
)
from align_system.prompt_engineering.outlines_prompts import (
    detailed_unstructured_treatment_action_text,
    detailed_unstructured_tagging_action_text
)
from align_system.utils import logging

from CybORG.Shared.Actions import (Sleep, Monitor, Analyse, Misinform, Remove, Restore)

log = logging.getLogger(__name__)

def format_choices(choices, available_actions, scenario_state):
    """
    If choices are not unique, format choices to include state information.
    """
    if len(set(choices)) != len(choices):
        log.warning("Unstructured text for available actions is not "
                    "unique, appending action parameters to choices")

        character_id_to_name = {c.id: c.name for c in scenario_state.characters}
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = []
        for a in available_actions:
            if(a.action_type == ActionTypeEnum.APPLY_TREATMENT
                and a.parameters is not None and len(a.parameters) > 0):
                choices.append(detailed_unstructured_treatment_action_text(a, character_id_to_name))
            elif(a.action_type == ActionTypeEnum.TAG_CHARACTER
                    and a.parameters is not None and len(a.parameters) > 0):
                choices.append(detailed_unstructured_tagging_action_text(a, character_id_to_name))
            else:
                # Not covering every possible case here, may need
                # to add more dedicated detailed prompts
                choices.append(a.unstructured)

    return choices


cage_action_mapping = {
        Sleep: "Sleep",
            Monitor: "Monitor",
            Analyse: "Analyse",
            Misinform: "Misinform",
            Remove: "Remove",
            Restore: "Restore"
        }

def cage_format_choices( available_actions): #, scenario_state):
    """
    Turn cage action classes into LLM-useful strings
    Extracted from: https://github.com/cage-challenge/cage-challenge-1/tree/main?tab=readme-ov-file#appendix-a---blue-action-sets
    """
    
    #hostnames = available_actions['hostname'].keys()
    choices = []
    for a in available_actions: #available_actions['action'].keys():
        choices.append(cage_action_mapping[a.cage_class])

    return choices

if __name__ == '__main__':
    choices = cage_format_choices({"hostname": {'test':True, 'test1':True },
        "action":{Sleep:True}})
    print(choices)
