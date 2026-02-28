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


class BaselineTaggingSystemPrompt:
    def __call__(self):
        return baseline_tagging_system_prompt()


@outlines.prompt
def start_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a shape representing the patients state \
based only on the provided description and the following protocol. \

1. Ambulatory Status
  * If the patient can walk: you must tag this patient as Line and stop protocol
  * If the patient cannot walk: you cannot tag this patient as Line, proceed to step 2
2. Breathing
  * If not breathing: open or reposition airway
      * Breathing does not resume: tag as Square and stop protocol
      * Breathing resumes: tag as Circle
  * If breathing:
      * Respiratory rate > 30 breaths/min: tag as Circle and stop protocol
      * Respiratory rate <= 30 breaths/min: proceed to step 3
3. Perfusion
  * Capillary refill > 2 seconds or no radial pulse: tag as Circle and stop protocol
  * Normal perfusion: proceed to step 4
4. Mental Status
  * Unable to follow simple commands: tag as Circle and stop protocol
  * Able to follow commands: tag as Triangle and stop protocol

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore injuries, symptoms or diagnosis.
3. Assign the appropriate shape for the provided patient state: Line, Triangle, Circle, Square.
2. Provide a brief explanation justifying the assigned tag shape, referencing relevant criteria.

If critical information is missing, select the most appropriate shape based on available details and clearly state \
the uncertainty.

"""


@outlines.prompt
def salt_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a shape representing the patients state \
based only on the provided description and the following protocol. \

1. Breathing Status:
  * If the patient is not breathing after airway repositioning: tag as Square and stop protocol
2. Major Injury Assessments
  * If the patient does not obey commands or cannot make purposeful movements: tag as Circle
      * If patient is unlikely to survive: tag as Square
  * If the patient does not have a peripheral pulse: tag as Circle
     * If patient is unlikely to survive: tag as Square
  * If the patient is in respiratory distress: Tag as Circle
     * If patient is unlikely to survive: tag as Square
  * If a major hemorrhages is not controlled: Tag as Circle
     * If patient is unlikely to survive: tag as Square
  * If no major injuries: proceed to Step 3
3. Other Injuries
  * If there are other serious injuries: tag as Triangle
  * If there are no injuries or only minor injuries: tag as Line

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore injuries, symptoms or diagnosis.
3. Assign the appropriate shape for the provided patient state: Line, Triangle, Circle, Square.
2. Provide a brief explanation justifying the assigned tag shape, referencing relevant criteria.

If critical information is missing, select the most appropriate shape based on available details and clearly state \
the uncertainty.
"""


@outlines.prompt
def bcd_tagging_system_prompt():
    """
You are working with an Army combat medic to help triage patients. \
Your role is to provide a shape representing the patients state \
based only on the provided description and the following protocol. \

1. Catastrophic Hemorrhage
  * If present: apply tourniquet or direct pressure immediately: tag as Circle and stop protocol
  * If absent: proceed to Step 2
2. Ambulatory Status
  * If the patient can walk: tag as Line and stop protocol
  * If the patient is unable to walk: proceed to Step 3
3. Breathing
  * If not breathing:
    * Open/reposition the airway
      * If still not breathing: tag as Square and stop protocol
      * If breathing resumes:  proceed to Step 4
  * If breathing is already present: proceed to Step 4
4. Level of consciousness
  * If not responsive to your voice: tag as Circle and stop protocol
  * If responds to your voice: proceed to Step 5
5. Respiratory Rate
  * If <12 or >23 breaths per minute: tag as Circle and stop protocol
  * If within normal range (12-23): proceed to Step 6
6. Circulatory Status
  * If pulse >100 bpm: tag as Circle and stop protocol
  * Else proceed to Step 7
7. Final Step
  * Tag as Triangle and stop protocol

Instructions:
1. Only evaluate the patient based on the criteria above. Ignore non hemorrhage injuries. Do not diagnose the patient.
3. Assign the appropriate shape for the provided patient state: Line, Triangle, Circle, Square.
2. Provide a brief explanation justifying the assigned tag shape, referencing relevant criteria.

If critical information is missing, select the most appropriate shape based on available details and clearly state \
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
