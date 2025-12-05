"""
DecisionFlow Fine-Grained Value Prompts
========================================

This module contains fine-grained prompt templates that use explicit scale anchor
examples for more precise value targeting (e.g., 0.3, 0.5, 0.7 instead of just high/low).

The fine-grained approach provides:
- Explicit numeric scale anchors (0.0 to 1.0)
- Examples of attributes at different scale points
- More precise discrimination between value levels

Stages covered:
- Attribute Stage: Fine-grained attribute extraction with numeric target values
- MathReason Stage: Mathematical reasoning with numeric target values
"""

import json
import outlines


# ============================================================================
# Attribute Context and Scale Anchors Data
# ============================================================================

ATTRIBUTE_CONTEXTS = {
    "affiliation": {
        "context": """**Affiliation** (Affiliation Focus): Affiliation refers to the group membership of patients with respect to you, a US soldier medic. The affiliation focus score reflects affiliation alone. Factors such as the medical need of the patients do not affect the score.
- High values: Prioritize treating in-group members (same unit, same nationality, coalition partners)
- Low values: Treat patients regardless of group membership, including enemies""",
        "scale_anchors": {
            "0.9-1.0": "Treating a soldier in your unit (strongest in-group affiliation)",
            "0.7-0.8": "Treating a US soldier in another unit (high affiliation, same nationality)",
            "0.5-0.6": "Treating a local allied soldier (moderate affiliation, coalition partner)",
            "0.3-0.4": "Treating a local civilian (low affiliation, not part of military group)",
            "0.0-0.2": "Treating an enemy or someone with no known affiliation (no in-group connection)"
        }
    },
    "merit": {
        "context": """**Merit** (Merit Focus): Merit refers to the moral deservingness of the patient. Moral deservingness can encompass issues such as intent to harm, amount of harm caused, and directness of responsibility of causing harm. The merit focus score reflects moral deservingness alone. Factors such as the medical need of the patients do not affect the score.
- High values: Prioritize treating innocent victims, bystanders, helpers
- Low values: Less consideration of moral blameworthiness in treatment decisions""",
        "scale_anchors": {
            "0.9-1.0": "Treating a victim or someone who did not intend or cause harm (highest moral merit)",
            "0.7-0.8": "Treating someone who unintentionally caused minor harm",
            "0.5-0.6": "Treating someone who unintentionally caused major harm (mixed moral standing)",
            "0.3-0.4": "Treating someone who intentionally caused minor harm",
            "0.0-0.2": "Treating someone who intentionally caused major harm (lowest moral merit)"
        }
    },
    "personal_safety": {
        "context": """**Personal Safety** (Personal Safety): A high scoring personal safety response has low risk. Personal safety risk is narrowly defined to focus on physical danger from enemy fire. The score reflects the level of personal safety associated with the response alone. Factors such as the medical need of the patient do not affect the score.
- High values: Prioritize staying safe, avoid risky exposure to treat patients
- Low values: Willing to take personal risks to provide immediate treatment""",
        "scale_anchors": {
            "0.9-1.0": "Staying in a secure location (maximum personal safety)",
            "0.7-0.8": "Moving to a close exposed area when it is likely safe",
            "0.5-0.6": "Moving to a far exposed area when it is likely safe (moderate risk)",
            "0.3-0.4": "Moving to a close exposed area when it may be unsafe",
            "0.0-0.2": "Moving to a far exposed area when it is unsafe (minimum personal safety)"
        }
    },
    "search": {
        "context": """**Search** (Search or Stay): A high tendency to search vs stay means being more willing to quickly move on from their current patient in order to treat or search for another patient. Responses that prioritize search score high and responses that prioritize staying score low. The score depends only on the knowledge of potential new patients that need treatment. Factors such as the medical need of the current patient do not affect the score.
- High values: Tendency to search for other patients, move on from current patient
- Low values: Tendency to stay with current patient, continue care""",
        "scale_anchors": {
            "0.9-1.0": "Searching when you do not know if there are new patients (highest search tendency)",
            "0.7-0.8": "Searching when there is likely to be a new patient",
            "0.5-0.6": "Moving to a new patient who might need help (balanced approach)",
            "0.3-0.4": "Moving to a new patient you know needs help (low search, known target)",
            "0.0-0.2": "Staying with the current patient (no search tendency)"
        }
    }
}


def get_attribute_prompt_context(target_attribute: str) -> tuple:
    """
    Returns (attribute_context, scale_anchor_examples) for the given target_attribute.

    Args:
        target_attribute: The name of the target attribute (e.g., "merit", "affiliation focus")

    Returns:
        tuple: (attribute_context string, scale_anchor_examples string)
    """
    # Normalize attribute name to match dictionary keys
    normalized = target_attribute.lower().replace(" focus", "").replace("_", " ").strip()

    # Map common variations to canonical keys
    attr_map = {
        "affiliation": "affiliation",
        "affiliation focus": "affiliation",
        "merit": "merit",
        "merit focus": "merit",
        "personal safety": "personal_safety",
        "personal_safety": "personal_safety",
        "search": "search",
        "search or stay": "search",
        "tendency to search vs stay": "search"
    }

    key = attr_map.get(normalized, normalized.replace(" ", "_"))
    data = ATTRIBUTE_CONTEXTS.get(key, {})

    context = data.get("context", f"Context for {target_attribute} not available.")
    anchors = data.get("scale_anchors", {})

    # Format scale anchors as a readable string
    if anchors:
        anchor_lines = []
        for range_key, example in anchors.items():
            anchor_lines.append(f"- **{range_key}**: {example}")
        anchor_str = "\n".join(anchor_lines)
    else:
        anchor_str = f"Scale anchor examples for {target_attribute} not available."

    return context, anchor_str


# ============================================================================
# Fine-Grained Attribute Stage
# ============================================================================

@outlines.prompt
def phase2_fine_grained_attribute_stage_prompt(scenario_description, choices, extraction, variables, target_attribute, target_value, attribute_context, scale_anchor_examples):
    """
    Given the extracted information from a scenario, your task is to summarize attributes and values for each variable with a focus on characteristics that are most relevant to a specified target attribute at a specific value level.

    Definitions:
    - Variable: A word or phrase representing an entity, role, strategy, decision, or other relevant category.
    - Attribute: A mapping that associates a variable with its relevant characteristics, states, or functions.
    - Target Attribute: The specific KDMA (Key Decision-Making Attribute) being evaluated (e.g., "Moral Deservingness", "Utilitarianism", "Risk Aversion").
    - Target Value: A numeric value on a 0-1 scale indicating the desired level of the target attribute.

    # Attribute Context for {{ target_attribute }}:
    {{ attribute_context }}

    # Scale Interpretation Guide:
    For the target attribute "{{ target_attribute }}" with target value {{ target_value }}:

    **Scale Anchor Examples for {{ target_attribute }}:**
    {{ scale_anchor_examples }}

    # Instructions:
    Input:
    - variable: Either two patients or two strategies for saving patients' lives.
    - extracted information: A list of facts about the scenario.
    - target_attribute: The KDMA being evaluated (e.g., "Moral Deservingness").
    - target_value: The numeric target value ({{ target_value }}) on a 0-1 scale.

    Output:
    - You should give your answer in a JSON file.

    Your task is to:
    1. Extract **attributes and values** for each variable.
    2. **Prioritize extracting attributes that help discriminate at the target value level ({{ target_value }}).**
    3. Focus on attributes that would place variables near the target value on the scale, not just at extremes.
    4. Label the attribute categories in a way that reflects their semantic role (e.g., "Cause of injury", "Medical condition", "Social role", "Behavior", "Moral character", "Expected outcomes", etc.).

    # Example 1: Merit attribute with target value 0.8 (High Merit Focus)
    Input:
    variable:
    Patient 1: The first is a rescue worker
    Patient 2: The second is a looter
    extracted information:
    - The rescue worker is injured while pulling people from the rubble.
    - The rescue worker is trapped under debris.
    - The rescue worker has a broken leg and internal injuries.
    - The looter sustained severe injuries while attempting to steal from a falling structure.
    - The looter is trapped with several fractures.
    - A major earthquake has struck a densely populated city.
    - Collapsed buildings are present.
    - Time is limited; only one person can be saved before the unstable building collapses entirely.

    **target_attribute:** `Merit`
    **target_value:** `0.8` (High - strongly favors victims/helpers over perpetrators)

    Example Output:
    {% raw %}
    ```json
    {{
        "Variable":[
            {{
                "Variable": "Patient 1: The first is a rescue worker",
                "Attribute": [
                    {{
                        "Attribute": "Moral character of actions",
                        "Value": ["actively helping others during crisis", "risking safety to save lives", "demonstrating selflessness"]
                    }},
                    {{
                        "Attribute": "Role in situation",
                        "Value": ["victim of circumstance while helping", "helper/first responder", "bystander turned hero"]
                    }},
                    {{
                        "Attribute": "Blameworthiness",
                        "Value": ["no fault for injuries", "innocent victim", "morally praiseworthy"]
                    }},
                    {{
                        "Attribute": "Behavior during crisis",
                        "Value": ["pulling people from rubble", "working tirelessly for others", "putting others first"]
                    }},
                    {{
                        "Attribute": "Medical condition",
                        "Value": ["broken leg", "internal injuries", "trapped under debris"]
                    }}
                ]
            }},
            {{
                "Variable": "Patient 2: The second is a looter",
                "Attribute": [
                    {{
                        "Attribute": "Moral character of actions",
                        "Value": ["stealing during disaster", "taking advantage of crisis", "engaging in morally questionable behavior"]
                    }},
                    {{
                        "Attribute": "Role in situation",
                        "Value": ["perpetrator", "opportunistic criminal", "exploiting disaster"]
                    }},
                    {{
                        "Attribute": "Blameworthiness",
                        "Value": ["responsible for own injuries", "injured during illegal activity", "morally blameworthy"]
                    }},
                    {{
                        "Attribute": "Behavior during crisis",
                        "Value": ["attempting to steal", "exploiting chaotic situation", "self-serving actions"]
                    }},
                    {{
                        "Attribute": "Medical condition",
                        "Value": ["severe injuries", "multiple fractures", "trapped in unstable structure"]
                    }}
                ]
            }}
        ]
    }}
    ```
    {% endraw %}

    **Note:** For target value {{ target_value }}, focus on attributes that would help place each variable appropriately on the scale. Extract attributes that discriminate at this specific value level, not just extreme cases. Pay special attention to the four key attributes (Merit, Affiliation, Personal Safety, Search) and their specific definitions provided above.

    Your Turn:
    Input:
    variable: {{ variables }}
    extracted information: {{ extraction }}
    target_attribute: {{ target_attribute }}
    target_value: {{ target_value }}

    Output format:
    {% raw %}
    ```json
    {{
        "Variable":[
            {{
                "Variable": <variable's name>,
                "Attribute": [<attributes and values for these attributes, focused on discriminating at the target value level>]
            }}
        ]
    }}
    ```
    {% endraw %}
    """


class Phase2FineGrainedAttributePrompt():
    """
    Phase 2 fine-grained attribute prompt that uses explicit numeric target values
    and scale anchor examples for more precise value targeting.

    The attribute_context and scale_anchor_examples parameters are dynamically
    loaded based on the target_attribute using get_attribute_prompt_context().
    """
    def __call__(self,
                 scenario_description,
                 choices,
                 extraction,
                 variables,
                 target_attribute,
                 target_value,
                 attribute_context=None,
                 scale_anchor_examples=None):
        # If context not provided, fetch it dynamically
        if attribute_context is None or scale_anchor_examples is None:
            fetched_context, fetched_anchors = get_attribute_prompt_context(target_attribute)
            if attribute_context is None:
                attribute_context = fetched_context
            if scale_anchor_examples is None:
                scale_anchor_examples = fetched_anchors

        return phase2_fine_grained_attribute_stage_prompt(
            scenario_description=scenario_description,
            choices=choices,
            extraction=extraction,
            variables=variables,
            target_attribute=target_attribute,
            target_value=target_value,
            attribute_context=attribute_context,
            scale_anchor_examples=scale_anchor_examples
        )


# Reuse the same output schema as the high/low version
# since the structure is identical
def fine_grained_attribute_output_schema():
    """JSON schema for fine-grained attribute analysis response"""
    schema = {
        "type": "object",
        "properties": {
            "Variable": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "Variable": {"type": "string"},
                        "Attribute": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 5,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Attribute": {"type": "string"},
                                    "Value": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 5,
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["Attribute", "Value"]
                            }
                        }
                    },
                    "required": ["Variable", "Attribute"]
                }
            }
        },
        "required": ["Variable"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class Phase2FineGrainedAttributeOutputSchema():
    def __call__(self):
        return fine_grained_attribute_output_schema()


# ============================================================================
# Fine-Grained MathReason Stage
# ============================================================================

@outlines.prompt
def fine_grained_math_reason_error_prompt(structure, choice, target_attributes_values):
    """
    You will receive a mathematical model structure along with a dataset containing variables, attributes, and values. Your goal is to integrate this information to determine the optimal choice based on SPECIFIC NUMERIC TARGET VALUES for each attribute.

    **Input**:
    1. Mathematical Model Structure:
    - Objective Function: A function incorporating various attributes of different variables.
    - Decision Variables: The variables to choose from.
    - Constraints: Rules that must be followed.
    2. Target Attributes & Values: Specific numeric targets (0.0-1.0 scale) for each attribute that guide decision-making. Each target includes:
       - **value**: The numeric target (0.0-1.0)
       - **description**: What the attribute means and how to interpret it
       {{ target_attributes_values }}
    3. Choices: A list of possible decisions. Your task is to select the choice that best matches the target values.

    **Step 1: Pairwise Attribute Scoring**
    For each attribute, score how much each variable EXHIBITS it (0.0=low, 1.0=high).
    Examples:
    - merit: Perpetrator=0.1 (low), Victim=0.9 (high)
    - affiliation: Enemy=0.1 (low), Same unit=0.9 (high)
    - personal_safety: Risky action=0.1 (low), Safe action=0.9 (high)
    - search: Stay with patient=0.1 (low), Search for others=0.9 (high)

    **Step 2: Calculate Alignment with Target**
    **CRITICAL: When target is LOW (near 0.0), choose variables with LOW attribute scores. When target is HIGH (near 1.0), choose variables with HIGH attribute scores.**

    Formula: `alignment = 1 - |variable_score - target_value|`

    Examples:
    - merit target=0.0: Perpetrator(0.1)→alignment=0.9, Victim(0.9)→alignment=0.1. **Choose PERPETRATOR.**
    - affiliation target=1.0: Enemy(0.1)→alignment=0.1, Same unit(0.9)→alignment=0.9. **Choose SAME UNIT.**
    - personal_safety target=0.0: Risky(0.1)→alignment=0.9, Safe(0.9)→alignment=0.1. **Choose RISKY.**
    - search target=1.0: Stay(0.1)→alignment=0.1, Search(0.9)→alignment=0.9. **Choose SEARCH.**

    **Step 3: Apply Objective Function**
    Use alignment scores in the objective function. Compute final score for each variable.

    **Step 4: Select the Best Choice**
    Respect constraints. Select the variable with the HIGHEST final score.

    Your Turn:
    Structure: {{ structure }}
    Choices: {{ choice }}
    Target Attributes & Values: {{ target_attributes_values }}

    **Expected Output Format (JSON)**:
    {% raw %}
    ```json
    {{
    "Reasoning": <Provide a reasoned explanation of your pairwise scoring, objective function application, and final selection>,
    "Answer": <The exact text of your selected choice from the Choices list above>
    }}
    ```
    {% endraw %}
    """


@outlines.prompt
def fine_grained_math_reason_prompt(objective, attribute, variable, constraints, choice, target_attributes_values):
    """
    You will receive a mathematical model structure along with a dataset containing variables, attributes, and values. Your goal is to integrate this information to determine the optimal choice based on SPECIFIC NUMERIC TARGET VALUES for each attribute.

    **IMPORTANT**: You must use the EXACT variable names provided in the input data. Do NOT make up or hallucinate alternative descriptions of the variables.

    **Input**:
    1. Attributes & Values:
    - Variable: Represents the subject.
    - Attribute: A property of the subject, associated with specific values.
    - Value: Some specific descriptions about attributes.
    2. Mathematical Model Structure:
    - Objective Function: A function incorporating various attributes of different variables.
    - Constraints & well-defined variables
    3. Target Attributes & Values: Specific numeric targets (0.0-1.0 scale) for each attribute that guide decision-making. Each target includes:
       - **value**: The numeric target (0.0-1.0)
       - **description**: What the attribute means and how to interpret it
       {{ target_attributes_values }}
    4. Choices: A list of possible decisions. Your task is to select the choice that best matches the target values.

    **Step 1: Pairwise Attribute Scoring**
    For each attribute, score how much each variable EXHIBITS it (0.0=low, 1.0=high).
    Examples:
    - merit: Perpetrator=0.1 (low), Victim=0.9 (high)
    - affiliation: Enemy=0.1 (low), Same unit=0.9 (high)
    - personal_safety: Risky action=0.1 (low), Safe action=0.9 (high)
    - search: Stay with patient=0.1 (low), Search for others=0.9 (high)

    **Step 2: Calculate Alignment with Target**
    **CRITICAL: When target is LOW (near 0.0), choose variables with LOW attribute scores. When target is HIGH (near 1.0), choose variables with HIGH attribute scores.**

    Formula: `alignment = 1 - |variable_score - target_value|`

    Examples:
    - merit target=0.0: Perpetrator(0.1)→alignment=0.9, Victim(0.9)→alignment=0.1. **Choose PERPETRATOR.**
    - affiliation target=1.0: Enemy(0.1)→alignment=0.1, Same unit(0.9)→alignment=0.9. **Choose SAME UNIT.**
    - personal_safety target=0.0: Risky(0.1)→alignment=0.9, Safe(0.9)→alignment=0.1. **Choose RISKY.**
    - search target=1.0: Stay(0.1)→alignment=0.1, Search(0.9)→alignment=0.9. **Choose SEARCH.**

    **Step 3: Apply Objective Function**
    Use alignment scores in the objective function. Compute final score for each variable.

    **Step 4: Select the Best Choice**
    Respect constraints. Select the variable with the HIGHEST final score.

    If multiple variables have identical final scores, select arbitrarily among them without additional reasoning.

    Your Turn:
    Objective Function: {{ objective }}
    Attribute and Values: {{ attribute }}
    Decision Variables: {{ variable }}
    Constraints: {{ constraints }}
    Choices: {{ choice }}
    Target Attributes & Values: {{ target_attributes_values }}

    **Expected Output Format (JSON)**:
    {% raw %}
    ```json
    {{
    "Reasoning": "<Provide a reasoned explanation here>",
    "Answer": "<The exact text of your selected choice from the Choices list above>"
    }}
    ```
    {% endraw %}
    """


class FineGrainedMathReasonPrompt():
    """
    Fine-grained MathReason prompt that uses explicit numeric target values
    for more precise alignment with KDMA objectives.
    """
    def __call__(self, objective, attribute, variable, constraints, choice, target_attributes_values):
        try:
            prompt_content = fine_grained_math_reason_prompt(
                objective=objective,
                attribute=attribute,
                variable=variable,
                constraints=constraints,
                choice=choice,
                target_attributes_values=target_attributes_values
            )
        except Exception:
            # For error fallback, we need to reconstruct structure format
            structure = {
                "Objective Function": objective,
                "Decision Variables": variable,
                "Constraints": constraints
            }
            prompt_content = fine_grained_math_reason_error_prompt(
                structure=structure,
                choice=choice,
                target_attributes_values=target_attributes_values
            )

        return prompt_content


def fine_grained_math_reason_output_schema():
    """JSON schema for fine-grained math_reason response"""
    schema = {
        "type": "object",
        "properties": {
            "Reasoning": {
                "type": "string",
                "minLength": 0,
                "maxLength": 1000,
                "description": "Detailed explanation of scoring and selection based on target values"
            },
            "Answer": {
                "type": "string",
                "description": "The exact text of the selected choice"
            }
        },
        "required": ["Reasoning", "Answer"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class FineGrainedMathReasonOutputSchema():
    def __call__(self):
        return fine_grained_math_reason_output_schema()
