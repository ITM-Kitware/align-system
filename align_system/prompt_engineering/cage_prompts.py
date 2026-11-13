"""Shared CIA prompts, rendered without an inference-library dependency."""

from functools import wraps
import inspect
import json
import re

import jinja2


def _prompt(function):
    """Preserve the whitespace and argument binding of the original templates."""
    text = inspect.cleandoc(function.__doc__)
    if function.__doc__.replace(' ', '').endswith('\n\n'):
        text += '\n'
    text = re.sub(r'(?![\r\n])(\b\s+)', ' ', text)
    environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                    keep_trailing_newline=True, undefined=jinja2.StrictUndefined)
    template = environment.from_string(text)
    signature = inspect.signature(function)

    @wraps(function)
    def render(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        return template.render(**arguments.arguments)

    return render


# Examples added from generated trajectories after seeded ones
# Each entry: label, first step, red actions, explanation, and the offset of the
# window shown for each supported window size. Blue is Monitor at every step.
_RESILIENCE_FEWSHOT_EXAMPLES = (
    ('B-line-C', 15, (
        'DiscoverNetworkServices on Auth0 (success: TRUE)',
        'ExploitRemoteService on Auth0 (success: TRUE)',
        'PrivilegeEscalate on Auth0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
    ), 'After finishing the chain on Auth0, it moves between Auth0 and Database0 with consecutive Impacts '
       'and never re-exploits Database0, which is B-line. Impacts on Database and none on Front indicate C.',
     {5: 1, 7: 1, 9: 0}),
    ('B-line-I', 21, (
        'Impact on Auth0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
    ), 'Every action is Impact, switching directly between Auth0 and Front0, which is B-line. '
       'Impacts on Front and none on Database indicate I.',
     {5: 2, 7: 1, 9: 0}),
    ('B-line-A', 23, (
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
        'Impact on Auth0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
    ), 'Every action is Impact, switching directly between target hosts, which is B-line. '
       'Impacts on both Database and Front indicate A.',
     {5: 1, 7: 0, 9: 0}),
    ('Meander-C', 34, (
        'ExploitRemoteService on Op_Server0 (success: TRUE)',
        'PrivilegeEscalate on Op_Server0 (success: TRUE)',
        'ExploitRemoteService on Op_Host1 (success: TRUE)',
        'PrivilegeEscalate on Op_Host1 (success: TRUE)',
        'ExploitRemoteService on User0 (success: TRUE)',
        'PrivilegeEscalate on User0 (success: TRUE)',
        'ExploitRemoteService on Database0 (success: TRUE)',
        'PrivilegeEscalate on Database0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
    ), 'It exploits non-target hosts, including User0, late in the episode before reaching its target, '
       'which is Meander. An Impact on Database and none on Front indicate C.',
     {5: 4, 7: 2, 9: 0}),
    ('Meander-I', 59, (
        'PrivilegeEscalate on Enterprise0 (success: TRUE)',
        'ExploitRemoteService on Enterprise2 (success: TRUE)',
        'PrivilegeEscalate on Enterprise2 (success: TRUE)',
        'ExploitRemoteService on None (success: FALSE)',
        'ExploitRemoteService on Op_Host0 (success: TRUE)',
        'PrivilegeEscalate on Op_Host0 (success: TRUE)',
        'ExploitRemoteService on Front0 (success: TRUE)',
        'PrivilegeEscalate on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
    ), 'It attacks non-target hosts, including Op_Host0, late in the episode before reaching its target, '
       'which is Meander. An Impact on Front and none on Database indicate I.',
     {5: 4, 7: 2, 9: 0}),
    ('Meander-A', 36, (
        'PrivilegeEscalate on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
        'DiscoverNetworkServices on Database0 (success: TRUE)',
        'ExploitRemoteService on User3 (success: TRUE)',
        'PrivilegeEscalate on User3 (success: TRUE)',
        'ExploitRemoteService on Database0 (success: TRUE)',
        'PrivilegeEscalate on Database0 (success: TRUE)',
        'Impact on Database0 (success: TRUE)',
    ), 'After impacting Front0, it rebuilds access to Database0 and exploits User3 in between, which is Meander. '
       'Impacts on both Front and Database indicate A.',
     {7: 2, 9: 0}),
    # No 5-step Meander-A window shows both an A target pair and off-path exploration.
    ('Meander-A', 65, (
        'Impact on Database0 (success: TRUE)',
        'DiscoverNetworkServices on Front0 (success: TRUE)',
        'ExploitRemoteService on Front0 (success: TRUE)',
        'PrivilegeEscalate on Front0 (success: TRUE)',
        'Impact on Front0 (success: TRUE)',
    ), 'It rebuilds access to Front0 at step 66, long after B-line finishes its first attack on each target, '
       'which is Meander. Impacts on both Database and Front indicate A.',
     {5: 0}),
)


def _format_resilience_fewshot_examples(window_size):
    blocks = []
    for label, first_step, red_actions, why, offsets in _RESILIENCE_FEWSHOT_EXAMPLES:
        if window_size not in offsets:
            continue
        offset = offsets[window_size]
        lines = [f'## Example {len(blocks) + 1}']
        for step, red_action in enumerate(red_actions[offset:offset + window_size],
                                          start=first_step + offset):
            lines += [f'Step {step}:', '  Blue: Monitor (success: TRUE)', f'  Red: {red_action}']
        lines += [f'Correct label: {label}', f'Why: {why}']
        blocks.append('\n'.join(lines))
    if not blocks:
        raise ValueError(f'No few-shot examples for {window_size}-step windows; use 5, 7, or 9')
    return '\n\n'.join(blocks)


@_prompt
def llms_for_resilience_red_agent_classification_fewshot_prompt(window_size, examples):
    """
    # DESCRIPTION

    You are an expert in cyber security defending an enterprise network. Analyze the network observations and choose the correct red agent from the list of possible red agents based on their description.

    Under # OBSERVATIONS, you will receive a sequence of network observations or trajectory actions, ordered from oldest to newest. Classify the attacker using the supplied evidence.

    # RED AGENT ATTACKERS

    ## RESILIENCE-TARGETING ATTACKER BEHAVIORS

    The six CIA classes use two strategies, described below:

    B-line strategy: Follows one direct route to its CIA targets. It works on one host at a time, in order (DiscoverNetworkServices, then ExploitRemoteService, then PrivilegeEscalate), without mixing in actions on other hosts. It progresses through intermediate hosts toward its targets. The first time it attacks a CIA target, it runs that chain on the target and then impacts it. It keeps its access afterwards: when it switches to a target it has already escalated, its next action is Impact on that target, with no new discovery, exploitation, or escalation. Once access to its targets is established, a B-line attacker focuses on Impact, often moving from one target host to another on consecutive actions.

    Meander strategy: Explores the network broadly in random order. It exploits and escalates many hosts outside its CIA target set, including User, Enterprise, Op_Host, and Op_Server hosts, and it returns to hosts it has already compromised. Some of its exploits fail with target None. After escalating an eligible target, it may impact that target one or more times. Before it impacts a different target, it rebuilds access to that target (service discovery, exploitation, and escalation), even when it compromised the host earlier, and it usually exploits other hosts in between. Impact is a small share of its actions; most of its actions are ExploitRemoteService and PrivilegeEscalate.

    ## RED-AGENT CLASSES

    - B-line-C: Uses the B-line strategy to reach and repeatedly impact Confidentiality-associated hosts whose names begin with Auth or Database.
    - B-line-I: Uses the B-line strategy to reach and repeatedly impact Integrity-associated hosts whose names begin with Auth or Front.
    - B-line-A: Uses the B-line strategy to reach and repeatedly impact Availability-associated hosts whose names begin with Auth, Front, or Database.
    - Meander-C: Uses the Meander strategy to impact Confidentiality-associated hosts whose names begin with Auth or Database.
    - Meander-I: Uses the Meander strategy to impact Integrity-associated hosts whose names begin with Auth or Front.
    - Meander-A: Uses the Meander strategy to impact Availability-associated hosts whose names begin with Auth, Front, or Database.

    ## TELLING B-LINE FROM MEANDER

    - Impact on one host followed directly by Impact on a different host indicates B-line. Meander never does this, because it rebuilds access before impacting a new target.
    - Exploitation and escalation that move between several different hosts, or that are mixed in before an Impact, indicate Meander. B-line finishes one host before starting the next.
    - Repeated Impact on a single host, or a single discover-exploit-escalate-Impact chain on one target, fits both strategies. Decide from the other actions in the window.

    ## TELLING C, I, AND A APART

    The CIA Impact target sets overlap: Auth is eligible for C, I, and A; Front for I and A; Database for C and A. Use the hosts that receive Impact. Meander also exploits Database or Front hosts that it is not allowed to impact, so exploitation alone does not establish the CIA target set.
    - Impacts on both Database and Front in the same window indicate A, since only A targets both.
    - Impacts on Database, with or without Auth, and none on Front indicate C. A is possible but less likely.
    - Impacts on Front, with or without Auth, and none on Database indicate I. A is possible but less likely.
    - Impacts only on Auth, or no Impact at all, are inconclusive about C, I, and A. Select the best-supported available class even when the evidence is ambiguous.

    # ENVIRONMENT RULES

    ## NETWORK STRUCTURE
    - 3 subnets: User subnetwork, Enterprise subnet with servers and the Defender host, and Operational subnet with Database, Front, and Auth servers and Operational Hosts.
    - Hostnames include User0-User4, Enterprise0-Enterprise2, Defender, Op_Host0-Op_Host2, Op_Server0, Auth, Database, and Front. Service hostnames may have numbered suffixes, such as Auth0.

    ## DEFENSE SETUP
    - 1 Network Defender in the Enterprise subnet.
    - Red team starts in the User subnet and maintains a foothold there.

    # OBSERVATION STRUCTURE

    Defender tables report detected Activity (Scan, Exploit, or None) and Compromised status (No, Unknown, User, or Privileged) for each host. Existing compromise may persist across observations.

    Trajectory logs list Blue and/or Red actions, their target host or subnet, and success status at each step. Blue actions describe the defender; Red actions describe the attacker. Failed Red actions still provide evidence about attempted behavior.

    Consider the sequence and target choices together. Infer the strategy from the actions in the window, without inventing earlier actions that are absent from it.

    {% if examples %}
    # EXAMPLES

    Each example is a {{ window_size }}-step window in the same format as # OBSERVATIONS, followed by its correct label.

    {{ examples }}

    {% endif %}
    # OUTPUT FORMAT

    Return a JSON object matching the supplied output schema:
    - "action_choice": exactly one of "B-line-C", "B-line-I", "B-line-A", "Meander-C", "Meander-I", or "Meander-A".

    Return only this single-field JSON object, without reasoning, explanations, Markdown, or additional fields.
    """


def llms_for_hybrid_cia_fewshot_prompt(window_size=9):
    """Keep live CIA policy selection synchronized with the few-shot classifier.

    The runner maps the returned attacker label to its matching PPO policy.
    Examples and classification instructions have a single source of truth.
    """
    return llms_for_resilience_red_agent_classification_fewshot_prompt(
        window_size=window_size,
        examples=_format_resilience_fewshot_examples(window_size))


class CAGERedAgentClassificationSchema():
    def __call__(self, choices):
        return json.dumps({
            "type": "object",
            "properties": {
                "action_choice": {
                    "type": "string",
                    "enum": list(choices),
                    "maxLength": max(map(len, choices)),
                },
            },
            "required": ["action_choice"],
            "additionalProperties": False,
        })


class CAGEHybridCIAFewshotSystemPrompt():
    def __init__(self, window_size=9):
        self.window_size = window_size

    def __call__(self):
        return llms_for_hybrid_cia_fewshot_prompt(self.window_size)


@_prompt
def cage_red_agent_classification_prompt(scenario_description, choices):
    """
    # OBSERVATIONS
    {{ scenario_description }}

    # AVAILABLE LABELS
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    Classify the Red agent using the supplied sequence. Return only a JSON object with one key, "action_choice", selecting one of the available labels. Do not include reasoning, an explanation, or additional fields.
    """


class CAGERedAgentClassificationPrompt():
    def __call__(self, scenario_description, choices):
        return cage_red_agent_classification_prompt(scenario_description, choices)
