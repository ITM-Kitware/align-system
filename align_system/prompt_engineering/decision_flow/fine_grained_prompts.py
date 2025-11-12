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
# Fine-Grained Attribute Stage
# ============================================================================

@outlines.prompt
def phase2_fine_grained_attribute_stage_prompt(scenario_description, choices, extraction, variables, target_attribute, target_value):
    """
    Given the extracted information from a scenario, your task is to summarize attributes and values for each variable with a focus on characteristics that are most relevant to a specified target attribute at a specific value level.

    Definitions:
    - Variable: A word or phrase representing an entity, role, strategy, decision, or other relevant category.
    - Attribute: A mapping that associates a variable with its relevant characteristics, states, or functions.
    - Target Attribute: The specific KDMA (Key Decision-Making Attribute) being evaluated (e.g., "Moral Deservingness", "Utilitarianism", "Risk Aversion").
    - Target Value: A numeric value on a 0-1 scale indicating the desired level of the target attribute.

    # Attribute Context:
    The following attributes have specific meanings in the medical triage context:

    **Merit** (Merit Focus): Favoring victims or bystanders over those who are perpetrators or otherwise potentially blame-worthy
    - High values: Prioritize treating innocent victims, bystanders, helpers
    - Low values: Less consideration of moral blameworthiness in treatment decisions

    **Affiliation** (Affiliation Focus): Favoring casualties who are members of your same group (family, military unit, coalition, nationality, region) over those who are not part of your group
    - High values: Prioritize treating in-group members (same unit, nationality, coalition)
    - Low values: Treat all patients equally regardless of group membership

    **Personal Safety** (Personal Safety): Tendency to hold back from moving to treat a patient when that requires risking exposure to enemy fire, waiting rather than immediately moving to treat the patient
    - High values: Prioritize personal safety, avoid risky exposure to treat patients
    - Low values: Willing to take personal risks to provide immediate treatment

    **Search** (Search or Stay): Tendency to move on from current patient to treat or search for another patient
    - High values: Tendency to search for other patients, move on from current patient
    - Low values: Tendency to stay with current patient, continue care

    # Scale Interpretation Guide:
    For the target attribute "{{ target_attribute }}" with target value {{ target_value }}:

    **Scale Anchor Examples for {{ target_attribute }}:**
    - **0.0-0.2 (Very Low)**: Attributes that strongly oppose or contradict the target attribute
      Merit: Perpetrators, those causing intentional harm, morally reprehensible actions
      Affiliation: Enemy combatants, adversary groups, non-affiliated individuals
      Personal Safety: Immediate action despite extreme danger, no hesitation under heavy fire
      Search: Committed to current patient, refuses to leave regardless of circumstances

    - **0.3-0.4 (Low)**: Attributes that somewhat contradict or weakly support the target attribute
      Merit: Individuals with minor moral failings, partially responsible for situation
      Affiliation: Allied but different units, coalition partners from different nations
      Personal Safety: Willing to take moderate risks, some caution in dangerous situations
      Search: Generally stays with current patient but considers other options

    - **0.5-0.6 (Moderate)**: Attributes that are neutral or show balanced characteristics
      Merit: Mixed moral standing, both positive and negative factors present
      Affiliation: Neutral parties, local civilians, ambiguous group membership
      Personal Safety: Balanced approach, weighs risks and benefits of exposure
      Search: Balances continuing current care with awareness of other patients

    - **0.7-0.8 (High)**: Attributes that strongly support the target attribute
      Merit: Victims, bystanders, those helping others, demonstrating moral merit
      Affiliation: Same military unit, same nationality, close coalition partners
      Personal Safety: Prioritizes avoiding risky exposure, waits for safer conditions
      Search: Actively looking for other patients, willing to move on from current patient

    - **0.9-1.0 (Very High)**: Attributes that exemplify the highest expression of the target attribute
      Merit: Selfless heroes, exceptional moral character, injured while saving others
      Affiliation: Immediate family, same squad/platoon, closest in-group members
      Personal Safety: Completely avoids any personal risk, refuses exposure regardless of patient need
      Search: Constantly moving on, searching for new patients, minimal time with each patient

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

    # Example 2: Affiliation attribute with target value 0.7 (High Affiliation Focus)
    For an affiliation-focused scenario with soldiers from different groups:
    - **High value (0.7-0.8)**: Extract attributes about same unit membership, nationality, coalition status
    - Focus on: "same squad", "same battalion", "US forces", "coalition partner" vs "local allied forces", "civilian"
    - Relevant attributes: "Group membership", "Military unit", "Nationality", "Coalition status"

    # Example 3: Personal Safety attribute with target value 0.3 (Low - willing to take risks)
    For a personal safety scenario involving risk exposure:
    - **Low value (0.3)**: Extract attributes showing willingness to accept risk for patient care
    - Focus on: "immediate movement despite danger", "exposed position", "under fire", "risk acceptance"
    - Relevant attributes: "Environmental danger", "Risk level", "Exposure to threats", "Response urgency"

    # Example 4: Search attribute with target value 0.6 (Moderate - balanced search vs stay)
    For a search/stay scenario with multiple patients:
    - **Moderate value (0.6)**: Extract attributes showing balanced consideration of current vs other patients
    - Focus on: "current patient stability", "awareness of other casualties", "treatment progress", "potential for other patients"
    - Relevant attributes: "Patient status", "Treatment completeness", "Other patients awareness", "Time with current patient"

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
    """
    def __call__(self,
                 scenario_description,
                 choices,
                 extraction,
                 variables,
                 target_attribute,
                 target_value):
        return phase2_fine_grained_attribute_stage_prompt(
            scenario_description=scenario_description,
            choices=choices,
            extraction=extraction,
            variables=variables,
            target_attribute=target_attribute,
            target_value=target_value
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
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "Variable": {"type": "string"},
                        "Attribute": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Attribute": {"type": "string"},
                                    "Value": {
                                        "type": "array",
                                        "minItems": 1,
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
    2. Target Attributes & Values: Specific numeric targets (0.0-1.0 scale) for each attribute that guide decision-making.
       {{ target_attributes_values }}
    3. Choices: A list of possible decisions, each indexed numerically. Your task is to determine the correct index based on the target values.

    **Step 1: Understand Target Values**
    For each attribute with a target value:
    - **0.0-0.2 (Very Low)**: Strongly oppose/contradict the attribute
    - **0.3-0.4 (Low)**: Somewhat contradict or weakly support the attribute
    - **0.5-0.6 (Moderate)**: Neutral or balanced expression of the attribute
    - **0.7-0.8 (High)**: Strongly support/express the attribute
    - **0.9-1.0 (Very High)**: Exemplify the highest expression of the attribute

    The objective is to select the variable whose characteristics BEST MATCH these target values.

    **Step 2: Score Each Variable**
    For each decision variable:
    - Based on the variable description and your understanding of the attributes, assign a score between 0 and 1 for each relevant attribute
    - Use Normal Distribution (Normal(μ, σ)) to represent uncertainty: Example: ~N(0.6, 0.1) if moderate confidence
    - Scores indicate how much that variable expresses each attribute

    **Step 3: Calculate Match with Target Values**
    For each variable:
    - For each attribute, compare the variable's score to the target value
    - Calculate the difference: |variable_score - target_value|
    - Compute expected value (mean) of distributions as the score
    - Smaller differences mean better alignment with targets

    **Step 4: Apply Objective Function & Select**
    - Use the objective function (if meaningful) to combine attribute scores
    - Select the variable with the smallest total difference from target values
    - The optimal choice is the variable that BEST MATCHES the target value profile
    - Respect all constraints (e.g., mutually exclusive choices)

    Your Turn:
    Structure: {{ structure }}
    Choices: {{ choice }}
    Target Attributes & Values: {{ target_attributes_values }}

    **Expected Output Format (JSON)**:
    {% raw %}
    ```json
    {{
    "Reasoning": <Explain your scoring for each variable, the differences from target values, and why you selected this choice>,
    "Answer": <Integer index corresponding to your final answer>
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
    3. Target Attributes & Values: Specific numeric targets (0.0-1.0 scale) for each attribute that guide decision-making.
       {{ target_attributes_values }}
    4. Choices: A list of possible decisions, each indexed numerically. Your task is to determine the correct index based on calculations.

    **Step 1: Understand Target Values**
    For each attribute with a target value:
    - **0.0-0.2 (Very Low)**: Strongly oppose/contradict the attribute
    - **0.3-0.4 (Low)**: Somewhat contradict or weakly support the attribute
    - **0.5-0.6 (Moderate)**: Neutral or balanced expression of the attribute
    - **0.7-0.8 (High)**: Strongly support/express the attribute
    - **0.9-1.0 (Very High)**: Exemplify the highest expression of the attribute

    The objective is to select the variable whose attribute values BEST MATCH these target values.

    **Step 2: Score Each Variable for Each Attribute**
    For each variable and each relevant attribute:
    - Examine the attribute values provided for that variable
    - Assign a score between 0 and 1 indicating how much that variable expresses the attribute
      - Use the attribute value descriptions to determine if the variable shows low, moderate, or high expression of the attribute
      - Example: For "affiliation" attribute:
        - "enemy combatant" → score ~0.0-0.1 (very low affiliation)
        - "coalition partner" → score ~0.5-0.6 (moderate affiliation)
        - "same unit" → score ~0.9-1.0 (very high affiliation)
    - Base your score on the ACTUAL attribute values provided, not on assumptions

    **Step 3: Calculate Match with Target Values**
    For each variable:
    - For each attribute, compare the variable's score to the target value
    - Calculate the difference: |variable_score - target_value|
    - Smaller differences mean better alignment with targets
    - You may also consider the objective function weighting if provided

    **Step 4: Apply Objective Function & Select**
    - Use the objective function (if meaningful) to combine attribute scores
    - Alternatively, select the variable with the smallest total difference from target values
    - The optimal choice is the variable that BEST MATCHES the target value profile
    - Respect all constraints (e.g., mutually exclusive choices)

    **Example**:
    {% raw %}
    If target values are: {"affiliation": 0.9, "merit": 0.3}
    And you have two variables:
    - Variable A: affiliation score 0.2, merit score 0.3
      → affiliation diff = |0.2 - 0.9| = 0.7, merit diff = |0.3 - 0.3| = 0.0, total = 0.7
    - Variable B: affiliation score 0.95, merit score 0.2
      → affiliation diff = |0.95 - 0.9| = 0.05, merit diff = |0.2 - 0.3| = 0.1, total = 0.15
    Variable B is the better match (total difference 0.15 vs 0.7)
    {% endraw %}

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
    "Reasoning": <Explain your scoring for each variable, the differences from target values, and why you selected this choice>,
    "Answer": <Integer index corresponding to your final answer>
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
                "minLength": 10,
                "maxLength": 2000,
                "description": "Detailed explanation of scoring and selection based on target values"
            },
            "Answer": {
                "type": "integer",
                "minimum": 0,
                "description": "Integer index of the selected choice"
            }
        },
        "required": ["Reasoning", "Answer"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class FineGrainedMathReasonOutputSchema():
    def __call__(self):
        return fine_grained_math_reason_output_schema()
