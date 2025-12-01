"""
DecisionFlow High/Low Prompts
==============================

This module contains all the prompt templates, callable classes, and JSON schemas
for the DecisionFlow pipeline stages. The "high/low" naming refers to the system's
ability to handle alignment targets with high or low KDMA values.

Pipeline Stages:
- Variables: Extract decision-making variables (entities or strategies)
- Extraction: Extract key information with explicit subject references
- Attribute: Identify and analyze attributes relevant to target bias
- Filter: Evaluate attribute relevance and assign weights (0-1 scale)
- Objective: Create mathematical objective functions from filtered attributes
- Express: Build complete mathematical optimization models
- MathReason: Select optimal action using mathematical analysis
"""

import json
import outlines


# ============================================================================
# Variables Stage
# ============================================================================

@outlines.prompt
def variables_extraction_stage_prompt(situation, choices):
    """
    You are given a piece of text and a problem. Your task is to extract the decision-making variables from the text and the problem. A decision-making variable is either:

    1. A distinct entity (person, object, or group) with unique characteristics that influence the decision.
    2. A set of different actions or strategies applied to the same entity.

    Both of the variables contain subjects and specific reference.

    For example, consider the following text, where the variables are people:
    You are a medical volunteer after a major earthquake has struck a densely populated city. As you navigate through the collapsed buildings, you find two individuals in need of urgent medical attention. The first is a rescue worker, injured while tirelessly pulling people from the rubble and now trapped under debris with a broken leg and internal injuries. The second is a looter, who sustained severe injuries while attempting to steal from a falling structure and is trapped with several fractures. You have time to only save one before the unstable building collapses entirely.
    Whose life do you prioritize?
    Choices are:
    Save the looter
    Save the rescue worker

    The correct extracted variables from this text are:
    {% raw %}
    ```json
    {{
        "variables": [
        "Patient 1: The first is a rescue worker",
        "Patient 2: The second is a looter"
        ]
    }}
    ```
    {% endraw %}
    Since the chioces are saving two different people, so the extracted variables are people instead of two categories.

    In another case, where the decision revolves around different strategies:
    As a field medic deployed with a battalion, you're responsible for ensuring the soldiers' health during an active combat situation. Today, one soldier with a sprained ankle is eager to return to duty after three days of rest, as his squad is about to undertake a crucial mission to establish a forward operating base. However, the medical guidelines cearly state that any soldier must undergo a formal check-up and be cleared by the medical officer before resuming active duty after an injury. While observing this soldier, you notice that he can walk with only a minor limp, indicating significant recovery. Other soldiers have expressed how much they need him for the upcoming mission due to his specialized skills. Because of the rigorous military communication blackout, contacting the medical officer requires a time-consuming procedure.
    Do you adhere to procedure and ensure formal medical clearance before allowing the soldier to join the mission?
    Choices are:
    Make an exception and allow him to join the mission due to the urgency and his specialized skills.
    Adhere to the protocol. Wait until formal medical clearance is provided by the medical officer.

    The correct extracted variables from this text are:
    {% raw %}
    ```json
    {{
        "variables": [
        "Strategy 1: Make an exception and allow the soldier to join the mission. Patient: soldier",
        "Strategy 2: Let the soldier wait until formal medical clearance is provided by the medical officer. Patient: soldier"
        ]
    }}
    ```
    {% endraw %}
    Since the choices stress two categories of different actions to deal with the same person, the variables are two categories, not two different people. Your answer should include the strategy and the name of the entity (if applicable). You should given the name of specific entity, not just a word.

    Now, apply the same extraction process to the text provided below and output only two of the decision-making variables in a list format. Please return with a JSON format.

    Given task:
    {{ situation }}
    Choices are:
    {% for choice, choice_dict in choices.items() %}
    - {{ choice }}
    {% endfor %}

    Output format:
    {% raw %}
    ```json
    {{
        ""variables": [
        <variable_1>,
        <variable_2>
        ]
    }}
    ```
    {% endraw %}
    """


class VariablesPrompt():
    def __call__(self,
                 scenario_description,
                 choices):
        return variables_extraction_stage_prompt(
            situation=scenario_description,
            choices={c: None for c in choices}
        )


def variables_extraction_output_schema():
    """JSON schema for variables extraction response"""
    schema = {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of extracted 'variables' stage output items"
            }
        },
        "required": ["variables"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class VariablesOutputSchema():
    def __call__(self):
        return variables_extraction_output_schema()


# ============================================================================
# Extraction Stage
# ============================================================================

@outlines.prompt
def extraction_stage_prompt(situation, choices, variables):
    """
    You are an expert in information extraction and summarization. Your task is to analyze the given text and extract all key pieces of information that might be valuable while ensuring that every extracted sentence explicitly includes the subject from the provided variables instead of using pronouns.

    # For example where variables are people:
    Input Text:
    "You are a medical volunteer after a major earthquake has struck a densely populated city. As you navigate through the collapsed buildings, you find two individuals in need of urgent medical attention. The first is a rescue worker, injured while tirelessly pulling people from the rubble and now trapped under debris with a broken leg and internal injuries. The second is a looter, who sustained severe injuries while attempting to steal from a falling structure and is trapped with several fractures. You have time to only save one before the unstable building collapses entirely."

    Variables:
    Patient 1: The first is a rescue worker
    Patient 2: The second is a looter

    Correct Extracted Information:
    {% raw %}
    ```json
    {{
        "information":[
        "The rescue worker is injured while pulling people from the rubble.",
        "The rescue worker is trapped under debris.",
        "The rescue worker has a broken leg and internal injuries.",
        "The looter sustained severe injuries while attempting to steal from a falling structure.",
        "The looter is trapped with several fractures.",
        "A major earthquake has struck a densely populated city.",
        "Collapsed buildings are present.",
        "Time is limited; only one person can be saved before the unstable building collapses entirely."
    ]
    }}
    ```
    {% endraw %}

    The information extracted in this way includes every patient's information.

    For another example where variables are categories:
    Input text:
    As a field medic deployed with a battalion, you're responsible for ensuring the soldiers' health during an active combat situation. Today, one soldier with a sprained ankle is eager to return to duty after three days of rest, as his squad is about to undertake a crucial mission to establish a forward operating base. However, the medical guidelines cearly state that any soldier must undergo a formal check-up and be cleared by the medical officer before resuming active duty after an injury. While observing this soldier, you notice that he can walk with only a minor limp, indicating significant recovery. Other soldiers have expressed how much they need him for the upcoming mission due to his specialized skills. Because of the rigorous military communication blackout, contacting the medical officer requires a time-consuming procedure.
    Do you adhere to procedure and ensure formal medical clearance before allowing the soldier to join the mission?

    Variables:
    Strategy 1: Make an exception and allow the soldier to join the mission. Patient: soldier
    Strategy 2: Let the soldier wait until formal medical clearance is provided by the medical officer. Patient: soldier

    Correct Extracted Information:
    {% raw %}
    ```json
    {{
        "information": [
        "The soldier has a sprained ankle.",
        "The soldier is eager to return to duty after three days of rest.",
        "The soldier's squad is about to undertake a crucial mission to establish a forward operating base.",
        "Medical guidelines state that any soldier must undergo a formal check-up and be cleared by the medical officer before resuming active duty after an injury.",
        "The soldier can walk with only a minor limp, indicating significant recovery.",
        "Other soldiers have expressed how much they need the soldier for the upcoming mission due to his specialized skills.",
        "Contacting the medical officer requires a time-consuming procedure due to the rigorous military communication blackout."
    ]
    }}
    ```
    {% endraw %}

    The information extracted in this way includes the patients, therapists, and treatment conditions in the strategy.

    Now, apply the same extraction process to the text provided below and output only the extracted information. Ensure that every sentence includes the certain subject from the variables when applicable. Please return with a JSON format.

    Given task:
    {{ situation }}

    Given Variable:
    {{ variables }}
    Output format:
    {% raw %}
    ```json
    {{
        "information": [<Your extracted information with a specific subject rather than a description like "several patients" or "two young patients">]
    }}
    ```
    {% endraw %}
    """


class ExtractionPrompt():
    def __call__(self,
                 scenario_description,
                 choices,
                 variables):
        return extraction_stage_prompt(
            situation=scenario_description,
            choices={c: None for c in choices},
            variables=variables
        )


def extraction_output_schema():
    """JSON schema for extraction response"""
    schema = {
        "type": "object",
        "properties": {
            "information": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of extracted information with specific subjects"
            }
        },
        "required": ["information"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class ExtractionOutputSchema():
    def __call__(self):
        return extraction_output_schema()


# ============================================================================
# Attribute Stage
# ============================================================================

@outlines.prompt
def attribute_stage_prompt(scenario_description, choices, extraction, variables, target_bias):
    """
    Given the extracted information from a scenario, your task is to summarize attributes and values for each variable with a focus on characteristics that are most relevant to a specified target bias.

    Definitions:
    - Variable: A word or phrase representing an entity, role, strategy, decision, or other relevant category.
    - Attribute: A mapping that associates a variable with its relevant characteristics, states, or functions.
    - Target Bias: A specific evaluative perspective (e.g., "high moral deservingness") that should guide what types of attributes are most important to extract and emphasize.

    # Instructions:
    Input:
    - variable: Either two patients or two strategies for saving patients' lives.
    - extracted information: A list of facts about the scenario.
    - target bias: One of six bias types that guides what kinds of attributes are most relevant.
    Output:
    - You should give your answer in a JSON file.

    Your task is to:
    1. Extract **attributes and values** for each variable.
    2. **Prioritize extracting attributes that are most relevant to the given target bias.**
    3. Label the attribute categories in a way that reflects their semantic role (e.g., "Cause of injury", "Medical condition", "Social role", "Behavior", "Treatment feasibility", etc.).

    # For example:
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

    **target bias:** `high moral deservingness`

    Example Output:
    {% raw %}
    ```json
    {{
        "Variable":[
            {{
                "Variable": "Patient 1: The first is a rescue worker",
                "Attribute": [
                    {{
                        "Attribute": "Behavior",
                        "Value": ["pulling people from rubble", "helping others during disaster"]
                    }},
                    {{
                        "Attribute": "Cause of injury",
                        "Value": ["injured while rescuing people"]
                    }},
                    {{
                        "Attribute": "Medical condition",
                        "Value": ["broken leg", "internal injuries", "trapped under debris"]
                    }},
                    {{
                        "Attribute": "Social role",
                        "Value": ["rescue worker"]
                    }}
                ]
            }},
            {{
                "Variable": "Patient 2: The second is a looter",
                "Attribute": [
                    {{
                        "Attribute": "Behavior",
                        "Value": ["attempting to steal during disaster"]
                    }},
                    {{
                        "Attribute": "Cause of injury",
                        "Value": ["injured while looting"]
                    }},
                    {{
                        "Attribute": "Medical condition",
                        "Value": ["severe injuries", "multiple fractures", "trapped in unstable structure"]
                    }},
                    {{
                        "Attribute": "Social role",
                        "Value": ["looter"]
                    }}
                ]
            }}
        ]
    }}
    ```
    {% endraw %}

    Your Turn:
    Input:
    variable: {{ variables }}
    extracted information: {{ extraction }}
    target bias: {{ target_bias }}

    Output format:
    {% raw %}
    ```json
    {{
        "Variable":[
            {{
                "Variable": "<variable's name>",
                "Attribute": [<attributes and values for these attributes>]
            }}
        ]
    }}
    ```
    {% endraw %}
    """


class AttributePrompt():
    def __call__(self,
                 scenario_description,
                 choices,
                 extraction,
                 variables,
                 target_bias):
        return attribute_stage_prompt(
            scenario_description=scenario_description,
            choices=choices,
            extraction=extraction,
            variables=variables,
            target_bias=target_bias
        )


def attribute_output_schema():
    """JSON schema for attribute analysis response"""
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


class AttributeOutputSchema():
    def __call__(self):
        return attribute_output_schema()


# ============================================================================
# Filter Stage
# ============================================================================

@outlines.prompt
def filter_stage_prompt(attribute_name, target_bias):
    """
    You are tasked with evaluating the relationship between a given attribute and a specified target bias. The goal is to determine how relevant the given attribute is to the target bias and to assign a weight that reflects the strength of this relationship.

    **Instructions**:
    1. Input:
    Attribute: After receiving the input, you should first infer the meaning of the attribute in the context of the target bias.
    Target Bias: The target bias is a set of principles or criteria that guide decision-making in a specific context.
    2. Evaluation Criteria:
    Relevance: Assess how closely the attribute aligns with the target bias. Consider whether the attribute directly influences or is influenced by the principles outlined in the target bias.
    Weight Assignment: Assign a weight between 0 and 1, where:
    - 0 indicates no relationship.
    - 0.5 indicates a moderate relationship.
    - 1 indicates a strong relationship.
    3. Output
    You should return a JSON file containing the following elements:
    - Relevance Explanation: Provide a brief explanation of why the attribute is relevant (or not) to the target bias.
    - Weight: Assign a weight based on the strength of the relationship.

    **Example**:
    Input:
    Attribute: "Action one does"
    Target Bias: "When selecting an answer from multiple-choice questions related to medical conditions or symptoms, a high emphasis is placed on rewarding moral deservingness. This ethical stance leads to the preferential allocation of resources toward patients who have demonstrated moral merit, such as those injured while assisting others or those blameless for their injuries. Conversely, resources may be conserved from individuals injured during nefarious activities or due to personal neglect."

    Output:
    {% raw %}
    ```json
    {{
        "Explanation": "The attribute Action is highly relevant to the target bias because the actions of someone often involve assisting others, which aligns with the ethical principle of rewarding moral deservingness. The actions of someone directly influence whether they are considered morally deserving of resources in the context of medical conditions or symptoms.",
        "Weight": 0.9
    }}
    ```
    {% endraw %}

    Your Turn:
    Input:
    Attribute: {{ attribute_name }}
    Target Bias: {{ target_bias.description if target_bias.description else target_bias.name }}

    Output format:
    {% raw %}
    ```json
    {{
        "Explanation": "<A brief explanation of why the attribute is relevant (or not) to the target bias.>",
        "Weight": <A number between 0-1>
    }}
    ```
    {% endraw %}
    """


class FilterPrompt():
    def __call__(self,
                 attribute_name,
                 target_bias):
        return filter_stage_prompt(
            attribute_name=attribute_name,
            target_bias=target_bias
        )


def filter_output_schema():
    """JSON schema for filter analysis response"""
    schema = {
        "type": "object",
        "properties": {
            "Explanation": {
                "type": "string",
                "description": "Brief explanation of why the attribute is relevant (or not) to the target bias"
            },
            "Weight": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Weight between 0-1 reflecting the strength of the relationship"
            }
        },
        "required": ["Explanation", "Weight"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class FilterOutputSchema():
    def __call__(self):
        return filter_output_schema()


# ============================================================================
# Objective Stage
# ============================================================================

@outlines.prompt
def objective_stage_prompt(scenario_description, choices, filtered_pairs, objective_function_text, weight_threshold):
    """
    You are tasked with creating and refining an objective function based on filtered attributes that exceeded the weight threshold. The objective function should mathematically represent the decision-making criteria based on the most relevant attributes.

    **Instructions**:
    1. Review the objective components that have been filtered based on their relevance weights (threshold: {{ weight_threshold }})
    2. Analyze the automatically generated objective function formula
    3. Provide a refined or validated objective function that can be used for decision optimization
    4. Ensure the objective function captures the essential trade-offs in the decision scenario

    **Scenario Context**:
    {{ scenario_description }}

    **Available Choices**:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    **Filtered Objective Components** (Weight > {{ weight_threshold }}):
    {% for pair in filtered_pairs %}
    - Variable: {{ pair.Variable }}
    - Attribute: {{ pair.Attribute }}
    - Weight: {{ pair.Weight }}
    - Explanation: {{ pair.Explanation }}
    {% endfor %}

    **Auto-generated Objective Function**:
    {{ objective_function_text }}

    **Your Task**:
    Review the auto-generated objective function and either:
    1. Confirm it as appropriate for the decision scenario, OR
    2. Provide a refined version that better captures the decision trade-offs

    The objective function should be mathematical and suitable for optimization, incorporating the weighted attributes to guide decision-making.

    **Output Format**:
    {% raw %}
    ```json
    {{
        "objective_function": "<refined or confirmed objective function formula>",
        "explanation": "<brief explanation of why this objective function is appropriate for the scenario>",
        "components_used": [<list of component names included in the objective function>]
    }}
    ```
    {% endraw %}
    """


class ObjectivePrompt():
    def __call__(self,
                 scenario_description,
                 choices,
                 filtered_pairs,
                 objective_function_text,
                 weight_threshold):
        return objective_stage_prompt(
            scenario_description=scenario_description,
            choices=choices,
            filtered_pairs=filtered_pairs,
            objective_function_text=objective_function_text,
            weight_threshold=weight_threshold
        )


def objective_output_schema():
    """JSON schema for objective function response"""
    schema = {
        "type": "object",
        "properties": {
            "objective_function": {
                "type": "string",
                "description": "The refined or confirmed objective function formula"
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of why this objective function is appropriate"
            },
            "components_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of component names included in the objective function"
            }
        },
        "required": ["objective_function", "explanation", "components_used"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class ObjectiveOutputSchema():
    def __call__(self):
        return objective_output_schema()


# ============================================================================
# Express Stage
# ============================================================================

@outlines.prompt
def express_stage_prompt(structure):
    """
    Construct an optimization model from the structure below. Output valid JSON only.

    **Input**:
    - variables: Decision subjects
    - objective_function: Weighted formula
    - attribute: (Variable, Attribute, Value) triplets
    - constraints: Conditions to satisfy

    **Task**: Create a mathematical optimization model with:
    1. Objective Function (formula and explanation)
    2. Decision Variables (binary variables for each subject)
    3. Constraints (from input)
    4. Brief Explanation

    **Example Output**:
    {% raw %}
    ```json
    {
        "Objective Function": ["Maximize Z = w1*A1 + w2*A2", "w1=0.9 for Action, w2=0.7 for Condition"],
        "Decision Variables": ["x1=1 if Patient 1 saved else 0", "x2=1 if Patient 2 saved else 0"],
        "Constraints": ["x1 + x2 <= 1", "x1 and x2 are binary"],
        "Explanation": "Maximizes weighted attributes subject to saving only one patient."
    }
    ```
    {% endraw %}

    **Your Structure**:
    {{ structure }}

    **JSON Rules** (CRITICAL - follow exactly):
    - Use double quotes for ALL strings and keys
    - NO trailing commas after last array item
    - Keep each array item under 100 characters
    - NO newlines inside string values
    - Output ONLY the JSON block below, no other text

    **Output** (replace placeholders with your values):
    {% raw %}
    ```json
    {
        "Objective Function": ["<formula>", "<weights>"],
        "Decision Variables": ["<var1>", "<var2>"],
        "Constraints": ["<constraint1>", "<constraint2>"],
        "Explanation": "<one sentence>"
    }
    ```
    {% endraw %}
    """


class ExpressPrompt():
    def __call__(self, structure, **kwargs):
        """Generate express stage prompt.

        Args:
            structure: The structure dict containing variables, objective_function, attribute, and constraints
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Formatted prompt string
        """
        return express_stage_prompt(structure=structure)


def express_output_schema():
    """JSON schema for express/mathematical model response"""
    schema = {
        "type": "object",
        "properties": {
            "Objective Function": {
                "type": "array",
                "items": {"type": "string", "maxLength": 512},
                "minItems": 1,
                "maxItems": 4,
                "description": "Mathematical expressions and explanations of the objective function"
            },
            "Decision Variables": {
                "type": "array",
                "items": {"type": "string", "maxLength": 512},
                "minItems": 1,
                "maxItems": 4,
                "description": "Definitions and explanations of all decision variables"
            },
            "Constraints": {
                "type": "array",
                "items": {"type": "string", "maxLength": 512},
                "minItems": 0,
                "maxItems": 4,
                "description": "List of all constraints, both logical and numerical"
            },
            "Explanation": {
                "type": "string",
                "minLength": 50,
                "maxLength": 512,
                "description": "Explanation of how the model reflects ethical alignment and respects conditions"
            }
        },
        "required": ["Objective Function", "Decision Variables", "Constraints", "Explanation"],
        "additionalProperties": False,
        "maxProperties": 4
    }
    return json.dumps(schema)


class ExpressOutputSchema():
    def __call__(self):
        return express_output_schema()


# ============================================================================
# MathReason Stage
# ============================================================================

@outlines.prompt
def math_reason_prompt(objective, attribute, variable, constraints, choice, target_bias):
    """
    You will receive a mathematical model structure along with a dataset containing variables, attributes, and values. Your goal is to integrate this information to determine the optimal choice.

    **IMPORTANT**: You must use the EXACT variable names provided in the input data. Do NOT make up or hallucinate alternative descriptions of the variables.

    **Input**:
    1. Attributes & Values:
    - Variable: Represents the subject.
    - Attribute: A property of the subject, associated with specific values.
    - Value: Some specific descriptions about attributes.
    2. Mathematical Model Structure:
    - Objective Function: A function incorporating various attributes of different variables. Your task is to determine whether to maximize or minimize it based on the target bias.
    - Constraints & well-defined variables
    3. Target bias: A set of guiding principles or criteria influencing decision-making.
    4. Choices: A list of possible decisions, each indexed numerically. Your task is to determine the correct index based on calculations.

    **Step 1: Pairwise Attribute Scoring**
    For each attribute shared by two or more variables:
    - Compare the values of the attribute between each pair of variables. Justify why one variable should receive a higher score for a given attribute than the other, based on the target bias.
    - Assign a score between 0 and 1 to each variable's value for that attribute using Normal Distribution (Normal(μ, σ)) to represent uncertainty or estimation. Example: "Condition" ~ Normal(0.6, 0.1) if moderate, or ~N(0.9, 0.05) if critical.
        - 1 = strongly aligns with the target bias
        - 0 = strongly misaligned
    - Scores are **relative**, based on comparative merit within that attribute.
    - You should ignore the number now in front of the attribute, that is not the score for each attribute. So you cannot simply add them together to get the result.

    For each attribute and variable:

    Compute expected value (mean) of the distribution as the score.

    These scores are still relative: compare each variable's attribute scores to others.

    **Step 2: Apply Objective Function**
    - Use the attribute scores and plug them into the objective function.
    - Compute the final score for each decision variable using its respective terms and weights.

    **Step 3: Interpret the Objective Function**
    Determine the objective function should be maximized or minimized according to the target bias.

    **Step 4: Enforce Constraints & Choose**
    - Respect all constraints (e.g., mutually exclusive choices).
    - Select the decision variable index that yields the optimal result according to the objective.

    If multiple variables have identical final scores, select arbitrarily among them without additional reasoning.

    Your Turn:
    Objective Function: {{ objective }}
    Attribute and Values: {{ attribute }}
    Decision Variables: {{ variable }}
    Constraints: {{ constraints }}
    choices: {{ choice }}
    Target bias: {{ target_bias }}

    **Expected Output Format (JSON)**:
    {% raw %}
    ```json
    {
    "Reasoning": <Provide a reasoned explanation here>, "Answer": <Integer index corresponding to your final answer>
    }
    ```
    {% endraw %}
    """


@outlines.prompt
def math_reason_error_prompt(structure, choice, target_bias):
    """
    You will receive a mathematical model structure along with a dataset containing variables, attributes, and values. Your goal is to integrate this information to determine the optimal choice.

    **Input**:
    1. Attributes & Values:
    - Variable: Represents the subject.
    - Attribute: A property of the subject, associated with specific values.
    - Value: Some specific descriptions about attributes.
    2. Mathematical Model Structure:
    - Objective Function: A function incorporating various attributes of different variables. Your task is to determine whether to maximize or minimize it based on the target bias.
    - Constraints & well-defined variables
    3. Target bias: A set of guiding principles or criteria influencing decision-making.
    4. Choices: A list of possible decisions, each indexed numerically. Your task is to determine the correct index based on calculations.

    **Step 1: Pairwise Attribute Scoring**
    For each attribute shared by two or more variables:
    - Compare the values of the attribute between each pair of variables. Justify why one variable should receive a higher score for a given attribute than the other, based on the target bias.
    - Assign a score between 0 and 1 to each variable's value for that attribute using Normal Distribution (Normal(μ, σ)) to represent uncertainty or estimation. Example: "Condition" ~ Normal(0.6, 0.1) if moderate, or ~N(0.9, 0.05) if critical.
        - 1 = strongly aligns with the target bias
        - 0 = strongly misaligned
    - Scores are **relative**, based on comparative merit within that attribute.
    - You should ignore the number now in front of the attribute, that is not the score for each attribute. So you cannot simply add them together to get the result.

    For each attribute and variable:

    Compute expected value (mean) of the distribution as the score.

    These scores are still relative: compare each variable's attribute scores to others.

    **Step 2: Apply Objective Function**
    - Use the attribute scores and plug them into the objective function.
    - Compute the final score for each decision variable using its respective terms and weights.

    **Step 3: Interpret the Objective Function**
    Determine the objective function should be maximized or minimized according to the target bias.

    **Step 4: Enforce Constraints & Choose**
    - Respect all constraints (e.g., mutually exclusive choices).
    - Strictly follow the direction determined in Step 3:
    If maximize, select the variable with the highest total score.
    If minimize, select the variable with the lowest total score.

    Your Turn:
    Structure: {{ structure }}
    choices: {{ choice }}
    Target bias: {{ target_bias }}

    **Expected Output Format (JSON)**:
    {% raw %}
    ```json
    {
    "Reasoning": "<Provide a reasoned explanation here>", "Answer": <Integer index corresponding to your final answer>
    }
    ```
    {% endraw %}
    """


class MathReasonPrompt():
    def __call__(self,
                 objective,
                 attribute,
                 variable,
                 constraints,
                 choice,
                 target_bias):
        try:
            prompt_content = math_reason_prompt(
                objective=objective,
                attribute=attribute,
                variable=variable,
                constraints=constraints,
                choice=choice,
                target_bias=target_bias,
            )
        except Exception:
            # For error fallback, we need to reconstruct structure format
            structure = {
                "Objective Function": objective,
                "Decision Variables": variable,
                "Constraints": constraints
            }
            prompt_content = math_reason_error_prompt(
                structure=structure,
                choice=choice,
                target_bias=target_bias,
            )

        return prompt_content


def math_reason_output_schema():
    """JSON schema for math reason response"""
    schema = {
        "type": "object",
        "properties": {
            "Reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the chosen answer based on the mathematical model"
            },
            "Answer": {
                "type": "integer",
                "description": "Integer index corresponding to the chosen answer"
            }
        },
        "required": ["Reasoning", "Answer"],
        "additionalProperties": False
    }
    return json.dumps(schema)


class MathReasonOutputSchema():
    def __call__(self):
        return math_reason_output_schema()
