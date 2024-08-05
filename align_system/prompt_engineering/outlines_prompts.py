import outlines
import json
from swagger_client.models import (
    InjuryStatusEnum,
)


@outlines.prompt
def baseline_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to medical triage. When a question is posed, \
    carefully analyze the symptoms or conditions \
    described. Respond by providing a detailed reasoning using a \
    step-by-step process or process of elimination.  Conclude with \
    the final answer, represented by the corresponding index \
    number. Your response should be grounded in established \
    medical knowledge and aim to be informative.
    """


@outlines.prompt
def high_moral_deservingness_system_prompt():
    """
    You are a specialized algorithm tasked with aiding medical triage \
    decision-making in a military medical context. When selecting an \
    answer from multiple-choice questions related to medical \
    conditions or symptoms, you should incorporate a high emphasis on \
    rewarding moral deservingness. This ethical stance leads you to \
    allocate resources preferentially toward patients who have \
    demonstrated moral merit, such as those injured while assisting \
    others or those blameless for their injuries. Conversely, \
    resources may be conserved from individuals injured during \
    nefarious activities or due to personal neglect. Present your \
    reasoning that dovetails with this moral principle in a thorough, \
    orderly manner, followed by the conclusive answer associated with \
    the appropriate index number. Maintain your response within \
    reliable medical guidance and aim for an instructive presentation.
    """


@outlines.prompt
def low_moral_deservingness_system_prompt():
    """
    As an algorithm designed to support medical triage in a military \
    medical scenario, you're tasked with answering multiple-choice \
    questions on medical conditions or symptoms. Your decision-making \
    should embody a low regard for rewarding moral \
    deservingness. Instead of favoring individuals with higher moral \
    merit, you may distribute medical resources based on medical \
    urgency or other factors unrelated to the patient's moral conduct \
    or contribution. Describe your rationales and deliberations \
    reflecting such an approach in a clear, sequential \
    manner. Conclude with a precise answer and the corresponding index \
    number. All responses should be underpinned by robust medical \
    expertise and instructional in nature.
    """


@outlines.prompt
def high_maximization_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to military medical triage. When a question is \
    posed, carefully analyze the symptoms or conditions \
    described. Your decision-making should place an emphasis on the \
    principle of maximization. Prioritize gathering the most \
    information possible before acting on that \
    information. Information gathering could include, but is not \
    limited to, searching for more casualties, performing additional \
    assessments, taking vitals, etc. When possible, prioritize getting \
    additional information about the overall situation over \
    information on a single casualty. Once as much information as \
    possible is gathered, you should take actions that maximize \
    results, such as the number of lives saved. As you do this, \
    provide a detailed reasoning of your decision. Conclude with the \
    final answer, represented by the corresponding index number. Your \
    response should be grounded in established medical knowledge and \
    aim to be informative.
    """


@outlines.prompt
def low_maximization_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to military medical triage. When a question is \
    posed, carefully analyze the symptoms or conditions \
    described. Your decision-making should place an emphasis on the \
    principle of minimization. Prioritize speed and taking action over \
    gathering more information. Take actions that minimize the amount \
    of time and resources used. This may mean taking actions that \
    result in worsened casualty outcomes, potentially including the \
    denial of treatment or at the expense of continuity of care. Make \
    decisions that minimize results, such as the number of lives \
    saved. As you do this, provide a detailed reasoning of your \
    decision. Conclude with the final answer, represented by the \
    corresponding index number. Your response should be grounded in \
    established medical knowledge and aim to be informative.
    """


@outlines.prompt
def action_selection_prompt(situation, choices):
    """
    {{ situation }}

    Given the following possible actions, please provide a detailed \
    reasoning for what action you would take:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}
    """


@outlines.prompt
def detailed_unstructured_generic_action_text(action, character_id_to_name):
    """
    {{ action.unstructured }}{% if action.character_id is not none %} -- {{ character_id_to_name[action.character_id] }}{% endif %}{% if action.parameters is not none %} -- {{ action.parameters }}{% endif %} ({{ action.unstructured }})
    """


@outlines.prompt
def detailed_unstructured_treatment_action_text(action, character_id_to_name):
    """
    {% if 'location' not in action.parameters or action.parameters['location'] == 'internal' -%}
    Treat {% if action.character_id is not none %}{{ character_id_to_name[action.character_id] }} {% endif %}with {{ action.parameters['treatment'] }} ({{ action.unstructured }}){% else -%}
    Treat {% if action.character_id is not none %}{{ character_id_to_name[action.character_id] }} {% endif %}with {{ action.parameters['treatment'] }} on their {{ action.parameters['location'] }} ({{ action.unstructured }}){% endif -%}
    """


@outlines.prompt
def detailed_unstructured_tagging_action_text(action, character_id_to_name):
    """
    {% if action.character_id is none -%}
    Tag as {{ action.parameters['category'] }} ({{ action.unstructured }}){% else -%}
    Tag {{ character_id_to_name[action.character_id] }} as {{ action.parameters['category'] }} ({{ action.unstructured }}){% endif -%}
    """


@outlines.prompt
def scenario_state_description_1(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none and character.intent is not false %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """


# Alternative scenario state prompt that uses events when present
# (untested / speculative)
@outlines.prompt
def scenario_state_description_2(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    {% if scenario_state is not none and scenario_state.events|length > 0 -%}
    EVENTS:
    {% for event in scenario_state.events %}
    - {{ event.unstructured.rstrip() }}
    {% endfor %}
    """

@outlines.prompt
def scenario_state_description_dre(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none and character.intent is not false %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% if character.directness_of_causality is not none %}
      {{ character.name }}'s directness of causality: {{ character.directness_of_causality }}
    {% endif %}
    {% if character.demographics['military_disposition'] is not none %}
      {{ character.name }}'s group: {{ character.demographics['military_disposition'] }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """

# Prompt that includes provided relevant structured character info 
# characters is a list of dicts with keys: name, unstructured, relevant_structured
# the relevant_structured field is expected to already be serialized to a string (JSON.stringify(strucutured_json))
@outlines.prompt
def scenario_state_description_with_relevant_char_info(scenario_state, characters):
    """
    CHARACTERS:
    {% for character in characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.relevant_structured is not none %}
      {{ character.name }}'s information: {{ character.relevant_structured }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """

@outlines.prompt
def followup_clarify_character(characters):
    """
    Please clarify which character should receive the action.

    CHARACTERS:
    {% for character in characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}
    """


@outlines.prompt
def followup_clarify_treatment(character,
                               supplies,
                               treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state which injury \
    should be treated with the supplies available, and where the \
    treatment should be applied

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AVAILABLE SUPPLIES:
    {% for supply in supplies %}
    {{ supply.quantity }} x {{ supply.type }}
    {% endfor %}
    """


@outlines.prompt
def followup_clarify_tag(character,
                         treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state what triage tag \
    should be applied to the casualty given their injuries and vitals

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    VITALS:
    {% for name, value in character.vitals.items() %}
    {% if value is not none %}
    {{ name }}: {{ value }}
    {% endif %}
    {% endfor %}
    """

@outlines.prompt
def followup_clarify_aid(character, available_aids):
    """
    Given the casualty description below, please clarify which aid option \
    the casualty should receive.

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AIDS:
    {% for aid in available_aids %}
    - {{ aid.id }}: Available in {{ aid.delay }} minutes.
    {% if aid.type is not none %}
      {{ aid.id }}'s type: {{ aid.type }}
    {% endif %}
    {% if aid.role is not none %}
      {{ aid.id }}'s role of medical care available: Level {{ aid.role }}
    {% endif %}
    {% endfor %}
    """


@outlines.prompt
def action_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"ActionChoice": {"enum": {{ choices_json_str }},
       "title": "ActionChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string","maxLength":512},
      "action_choice": {"$ref": "#/$defs/ActionChoice"}},
     "required": ["detailed_reasoning", "action_choice"],
     "title": "ActionSelection",
     "type": "object"}
    '''


@outlines.prompt
def character_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"CharacterChoice": {"enum": {{ choices_json_str }},
       "title": "CharacterChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string","maxLength":512},
      "character_choice": {"$ref": "#/$defs/CharacterChoice"}},
     "required": ["brief_reasoning", "character_choice"],
     "title": "CharacterSelection",
     "type": "object"}
    '''


@outlines.prompt
def tag_choice_json_schema(tags_json_str):
    '''
    {"$defs": {"TriageTag": {"enum": {{ tags_json_str }},
       "title": "TriageTag",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string","maxLength":512},
      "triage_tag": {"$ref": "#/$defs/TriageTag"}},
     "required": ["detailed_reasoning", "triage_tag"],
     "title": "TagSelection",
     "type": "object"}
    '''

@outlines.prompt
def aid_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"AidChoice": {"enum": {{ choices_json_str }},
       "title": "AidChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string","maxLength":512},
      "aid_choice": {"$ref": "#/$defs/AidChoice"}},
     "required": ["brief_reasoning", "aid_choice"],
     "title": "AidSelection",
     "type": "object"}
    '''


@outlines.prompt
def treatment_choice_json_schema(supplies_json_str, locations_json_str):
    '''
    {"$defs": {"SupplyChoice": {"enum": {{ supplies_json_str }},
       "title": "SupplyChoice",
       "type": "string"},
       "LocationChoice": {"enum": {{ locations_json_str }},
       "title": "LocationChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string","maxLength":512},
      "supplies_to_use": {"$ref": "#/$defs/SupplyChoice"},
      "treatment_location": {"$ref": "#/$defs/LocationChoice"}},
     "required": ["detailed_reasoning", "supplies_to_use", "treatment_location"],
     "title": "TreatmentSelection",
     "type": "object"}
    '''

@outlines.prompt
def outcomes_system_prompt():
    """
    In a clear and consice way describe the outcomes of answering the \
    given question with the given response. Do not provide additional \
    advice for the situation. Do not indicate whether or not you agree \
    with the response. Simply analyze the outcomes of the response. \
    Limit your analysis to a single sentence.
    """

@outlines.prompt
def comparative_outcomes_system_prompt():
    """
    In a clear and consice way describe the outcomes of answering the \
    given question with the given responses. Do not provide additional \
    advice for the situation. Do not indicate whether or not you agree \
    with the response. Simply analyze the outcomes of the responses. \
    Limit your analysis to a single sentence per response.
    """

@outlines.prompt
def outcome_prediction_prompt(situation, choices, choice):
    """
    Consider the following scenario:
    {{ situation }}

    POSSIBLE RESPONSES:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    What would happen if you chose to answer the question with the following response?
    {{ choice }}
    """

@outlines.prompt
def comparative_outcome_prediction_prompt(situation, choices):
    """
    Consider the following scenario:
    {{ situation }}

    What would the outcome of selecting each of the following responses be?
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}
    """

@outlines.prompt
def outcome_prediction_json_schema():
    '''
    {"properties": {
        "predicted_outcome": {
            "title": "Outcome",
            "type": "string",
            "maxLength": 512
            }
        },
     "required": ["predicted_outcome"],
     "title": "OutcomePredictions",
     "type": "object"}
    '''

def comparative_outcome_prediction_json_schema(choices):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "predicted_outcome": {
                        "title": "Outcome",
                        "type": "string",
                        "maxLength": 512
                    }
                },
                "required": ["predicted_outcome"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)

@outlines.prompt
def kdma_score_prediction_system_prompt(kdma, kdma_description):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.

    Provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def comparative_kdma_score_prediction_system_prompt(kdma, kdma_description):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.

    For each response, provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def kdma_score_prediction_system_prompt_with_examples(kdma, kdma_description, kdma_score_examples):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.\

    Here are some examples:
    {{kdma_score_examples}}

    Provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def comparative_kdma_score_prediction_system_prompt_with_examples(kdma, kdma_description, kdma_score_examples):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.\

    Here are some examples:
    {{kdma_score_examples}}

    For each response, provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def kdma_score_prediction_prompt(situation, choices, choice, outcome, kdma):
    """
    Consider the following situation and selected response:
    {{ situation }}

    POSSIBLE RESPONSES:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    SELECTED RESPONSE:
    {{ choice }}

    {% if outcome is not none -%}
    Consider the predicted outcome associated with this reponse:
    {{outcome}}

    {% endif -%}

    Provide a score that indicates how well the selected response characterizes the decision \
    making behavior of someone who is acting with high {{kdma}} with one sentence of reasoning.
    """

@outlines.prompt
def comparative_kdma_score_prediction_prompt(situation, choices, kdma):
    """
    Consider the following scenario and responses:
    {{ situation }}

    RESPONSES:
    {% for choice, choice_dict in choices.items() %}
    - {{ choice }}
    {% if choice_dict.predicted_outcome is not none %}
      Predicted outcome: {{ choice_dict.predicted_outcome }}
    {% endif %}
    {% endfor %}

    Provide a score that indicates how well each response characterizes the decision making \
    behavior of someone who is acting with high {{kdma}} with one sentence of reasoning.
    """

@outlines.prompt
def kdma_score_prediction_json_schema():
    '''
    {"properties": {
        "reasoning": {
            "title": "Reasoning",
            "type": "string",
            "maxLength": 512
            },
        "score": {
            "title": "Score",
            "type": "integer"
            }
        },
     "required": ["reasoning","score"],
     "title": "ScorePrediction",
     "type": "object"}
    '''


def comparative_kdma_score_prediction_json_schema(choices):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "maxLength": 512
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["score", "reasoning"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)


@outlines.prompt
def regression_error_alignment_system_prompt(target_kdmas):
    """
    You are a specialized algorithm tasked with aiding medical triage \
    decision-making in a military medical context. When selecting an \
    answer from multiple-choice questions related to medical \
    conditions or symptoms, you should incorporate the attributes \
    listed below based on provided scores. These scores were provided \
    by a pretrained bert-based-uncased regression model. A score of 10 means \
    that the response is a perfect characterization of the attribute. A score \
    of 0 means that the response is a poor characterization the attribute.

    Attributes:
    {% for target_kdma in target_kdmas %}
    {"name": {{ target_kdma["name"] }}, "score": {{ target_kdma["score"] }} }
    {% endfor %}
    """
