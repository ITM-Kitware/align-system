import outlines


@outlines.prompt
def tagging_scenario_state_description(scenario_state):
    """
PATIENT DESCRIPTION:
{{ scenario_state.unstructured.rstrip() }}
"""


class TaggingScenarioDescription():
    def __call__(self, scenario_state):
        return tagging_scenario_state_description(scenario_state)


@outlines.prompt
def tagging_selection_prompt(situation, choices):
    """
{{ situation }}

Given the following possible tags, please provide a detailed \
reasoning for which tag you would assign:
{% for choice in choices %}
- {{ choice }}
{% endfor %}"""


class TaggingSelectionPrompt():
    def __call__(self, scenario_description, choices):
        return tagging_selection_prompt(scenario_description, choices)


@outlines.prompt
def baseline_tagging_system_prompt():
    """
You are an army combat medic trained to classify patient conditions into four triage categories. Your task is to read \
the provided patient description and assign one of the four triage categories:
  * Red (Immediate) - Life-threatening injuries requiring immediate intervention. Patient is likely to survive if \
treated quickly.
  * Yellow (Delayed) - Serious but not life-threatening injuries. Treatment can be delayed without immediate risk.
  * Green (Minor) - Minor injuries.
  * Black (Expectant) - Chances of survival are low, even with treatment.

Instructions:
1. Evaluate the patient description.
2. Provide brief reasoning.
3. Assign only one triage category: Green, Yellow, Red, or Black.

If critical information is missing, select the most appropriate category based on available details and clearly state \
the uncertainty.
"""


class BaselineTaggingSystemPrompt():
    def __call__(self):
        return baseline_tagging_system_prompt()


@outlines.prompt
def start_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a color representing the patients state \
based only on the provided description and the following criteria. \

1. Ambulatory Status
  * If the patient can walk: tag as Green
  * If unknown or non-ambulatory: proceed to step 2
2. Breathing
  * If not breathing: open or reposition airway
      * Breathing does not resume: tag as Black
      * Breathing resumes: tag as Red
  * If breathing:
      * Respiratory rate > 30 breaths/min: tag as Red
      * Respiratory rate <= 30 breaths/min: proceed to step 3
3. Perfusion
  * Capillary refill > 2 seconds or no radial pulse: tag as Red
  * Normal perfusion: proceed to step 4
4. Mental Status
  * Unable to follow simple commands: tag as Red
  * Able to follow commands: tag as Yellow

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore any injuries, symptoms or diagnosis.
3. Assign the appropriate color for the provided patient state: Green, Yellow, Red, or Black.
2. Provide a brief explanation justifying the assigned tag color, referencing relevant criteria.

If critical information is missing, select the most appropriate color  based on available details and clearly state \
the uncertainty.

"""


@outlines.prompt
def salt_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a color representing the patients state \
based only on the provided description and the following criteria. \

1. Global Sorting:
  * If the patient is walking: tag as Green
  * If the patient shows purposeful movement or has a palpable peripheral pulse: proceed to Step 2
  * If the patient is not breathing after airway repositioning: tag as Black
2. Individual Assessment (for patients not tagged during global sorting)
  * If the patient does not obey commands or cannot make purposeful movements: tag as Red
      * If patient is unlikely to survive: tag as Black
  * If the patient does not have a peripheral pulse: tag as Red
     * If patient is unlikely to survive: tag as Black
  * If the patient is in respiratory distress: Tag as Red
     * If patient is unlikely to survive: tag as Black
  * If a major hemorrhages is not controlled: Tag as Red
     * If patient is unlikely to survive: tag as Black
3. Other Injuries
  * If there are other serious injuries: tag as Yellow
  * If there are other minor injuries: tag as Green

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore any symptoms or diagnosis.
3. Assign the appropriate color for the provided patient state: Green, Yellow, Red, or Black.
2. Provide a brief explanation justifying the assigned tag color, referencing relevant criteria.

If critical information is missing, select the most appropriate color  based on available details and clearly state \
the uncertainty.
"""


@outlines.prompt
def bcd_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a color representing the patients state \
based only on the provided description and the following criteria. \

1. Catastrophic Hemorrhage
  * If present: apply tourniquet or direct pressure immediately: tag as Red
  * If absent: proceed to Step 2
2. Ambulatory Status
  * If the patient can walk: tag as Green
  * If not: proceed to Step 3
3. Breathing
  * If not breathing:
    * Open/reposition the airway
      * If still not breathing: tag as Black
      * If breathing resumes:  proceed to Step 4
  * If breathing is already present: proceed to Step 4
4. Level of consciousness
  * If unconscious or unresponsive: tag as Red
  * If responsive: proceed to Step 5
5. Respiratory Rate
  * If <12 or >23 breaths per minute: tag as Red
  * If within normal range (12-23): proceed to Step 6
6. Circulatory Status
  * If pulse >100 bpm: tag as Red
  * If normal: proceed to Step 7
7. Final Determination
  * If none of the above critical conditions apply: tag as Yellow

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore any symptoms or diagnosis.
3. Assign the appropriate color for the provided patient state: Green, Yellow, Red, or Black.
2. Provide a brief explanation justifying the assigned tag color, referencing relevant criteria.

If critical information is missing, select the most appropriate color  based on available details and clearly state \
the uncertainty.
"""


class TaggingSystemPrompt:
    def __call__(self, target_kdma, target_value):
        if target_kdma == "START":
            return start_tagging_system_prompt()
        elif target_kdma == "SALT":
            return salt_tagging_system_prompt()
        elif target_kdma == "BCD_SIEVE":
            return bcd_tagging_system_prompt()
        else:
            raise ValueError(f"Unknown target tagging protocol: {target_kdma}")
