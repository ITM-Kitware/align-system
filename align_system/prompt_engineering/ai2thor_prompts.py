import json

from align_system.prompt_engineering.compat.outlines.templates import prompt as compat_outlines_prompt


@compat_outlines_prompt
def ai2thor_proposer_system_prompt(num_candidates, rollout_horizon):
    """
    You are an embodied planning model.
    Generate {{ num_candidates }} diverse candidate plans.
    - Each plan is 1 to {{ rollout_horizon }} actions.
    - Use ONLY the tool names provided.
    - Args MUST satisfy each tool schema.
    - IMPORTANT objectId rule: For tools requiring objectId (TeleportNearObject, PickupObject, \
OpenObject, CloseObject, ToggleObjectOn/Off), you MUST copy the exact full objectId string \
from the observation's visible lines (the value after 'id='). Never use object type names \
like 'Apple' as objectId. Full objectIds contain '|' characters.
    - Avoid repeating the same last action unless clearly helpful.
    """


class AI2ThorProposerSystemPrompt():
    def __call__(self, num_candidates, rollout_horizon):
        return ai2thor_proposer_system_prompt(
            num_candidates,
            rollout_horizon)


@compat_outlines_prompt
def ai2thor_proposer_prompt(task, tools, action_history, failed_attempts,
                            num_candidates):
    """
    Task: {{ task }}

    Available tools:
    {% for tool in tools %}
    - {{ tool.name }}: {{ tool.description }}
    {% endfor %}

    Action history:
    {% if action_history %}
    {% for action in action_history %}
    - {{ action.tool_name }}({{ action.args }})
    {% endfor %}
    {% else %}
    None
    {% endif %}
    {% if failed_attempts %}

    Failed attempts (these actions did NOT work; do not repeat them with the same args):
    {% for action in failed_attempts %}
    - {{ action.tool_name }}({{ action.args }})
    {% endfor %}
    {% endif %}

    Generate {{ num_candidates }} diverse candidate plans.
    """


class AI2ThorProposerPrompt():
    def __call__(self, scenario_state, tools, action_history, failed_attempts,
                 num_candidates):
        return ai2thor_proposer_prompt(
            scenario_state.unstructured,
            tools,
            action_history,
            failed_attempts,
            num_candidates)


def ai2thor_proposer_json_schema(tools,
                                 num_candidates,
                                 rollout_horizon,
                                 rationale_max_length=512):
    # Union of the argument properties declared across the tool
    # schemas; per-tool argument requirements can't be expressed here
    # without a much more complex (anyOf) schema
    arg_properties = {}
    for tool in tools:
        arg_properties.update((tool.json_schema or {}).get('properties', {}))

    json_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": num_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": rollout_horizon,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool_name": {
                                        "type": "string",
                                        "enum": [t.name for t in tools]
                                    },
                                    "args": {
                                        "type": "object",
                                        "properties": arg_properties,
                                    },
                                },
                                "required": ["tool_name", "args"]
                            }
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                            **({"maxLength": rationale_max_length}
                               if rationale_max_length > 0 else {})
                        },
                    },
                    "required": ["actions", "rationale"]
                }
            }
        },
        "required": ["candidates"]
    }
    return json.dumps(json_schema)


class AI2ThorProposerSchema():
    def __init__(self, rationale_max_length=512):
        self.rationale_max_length = rationale_max_length

    def __call__(self, tools, num_candidates, rollout_horizon):
        return ai2thor_proposer_json_schema(
            tools,
            num_candidates,
            rollout_horizon,
            self.rationale_max_length)
