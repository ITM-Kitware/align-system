"""Pure action resolution and views shared by open-world agent workflows.

Resolution uses an isolated snapshot of the current offers and local bookkeeping.
It cannot refresh the environment, invoke tools, or submit an action. Reviewer
payloads must use the sanitized view helpers, never serialize the raw snapshot.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json

from swagger_client.models import ActionTypeEnum

from align_system.data_models.compat.ta3_ph1_client_models import (
    CharacterTagEnum, InjuryLocationEnum)
from align_system.utils import get_swagger_class_enum_values, logging


log = logging.getLogger(__name__)

TRIAGE_TAGS = get_swagger_class_enum_values(CharacterTagEnum)
INJURY_LOCATIONS = get_swagger_class_enum_values(InjuryLocationEnum)

# Missing-value markers are not treatment names, even when inventory is unknown.
TREATMENT_PLACEHOLDERS = frozenset({
    'null', 'none', 'n/a', 'na', 'unknown', 'unspecified', 'undefined', 'nan',
})

# Action types the (live) environment rejects without a character_id
CHARACTER_REQUIRED_ACTIONS = {
    ActionTypeEnum.CHECK_VITALS,
    ActionTypeEnum.TREAT_PATIENT,
    ActionTypeEnum.MOVE_TO,
    ActionTypeEnum.MOVE_TO_EVAC,
    ActionTypeEnum.TAG_CHARACTER,
}


# The agent-facing tool name for each action type in the pipeline
TOOL_NAMES_BY_ACTION_TYPE = {
    ActionTypeEnum.CHECK_VITALS: 'check_vitals',
    ActionTypeEnum.TREAT_PATIENT: 'treat_patient',
    ActionTypeEnum.TAG_CHARACTER: 'tag_character',
    ActionTypeEnum.MOVE_TO: 'move_to',
    ActionTypeEnum.MOVE_TO_EVAC: 'move_to_evac',
    ActionTypeEnum.SEARCH: 'search',
    ActionTypeEnum.MESSAGE: 'send_message',
    ActionTypeEnum.END_SCENE: 'end_scene',
}


NATIVE_ACTION_TOOLS = frozenset(TOOL_NAMES_BY_ACTION_TYPE.values())


def review_action_tool_schemas(schemas):
    """Copy native JSON schemas, exposing resolver locations only to reviewers.

    Native tools retain their existing signatures and normalization behavior.
    Restrict enrichment to an unconstrained string property: custom constraints
    and references must remain intact rather than be widened or reinterpreted.
    """
    result = deepcopy(schemas)
    treatment = result.get('treat_patient')
    if not isinstance(treatment, dict):
        return result
    properties = treatment.get('properties')
    if not isinstance(properties, dict):
        return result
    location = properties.get('injury_location')
    annotations = {'type', 'title', 'description', 'default', 'examples'}
    if (not isinstance(location, dict) or location.get('type') != 'string'
            or not set(location) <= annotations):
        return result
    location['enum'] = ['', *INJURY_LOCATIONS]
    guidance = (
        'Select a listed canonical location; use "unspecified" when the location '
        'or laterality is unclear. Omitting this argument or passing an empty '
        'string uses the offered action location, otherwise "unspecified".')
    location['description'] = ' '.join(filter(None, (
        location.get('description'), guidance)))
    return result


def compact_action_catalog(schemas):
    """Remove titles only at known schema nodes, never in literal JSON data."""
    def compact(schema):
        result = deepcopy(schema)
        if not isinstance(result, dict):
            return result
        result.pop('title', None)
        for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
            if isinstance(result.get(key), dict):
                result[key] = {name: compact(value) for name, value in result[key].items()}
        for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
            if isinstance(result.get(key), list):
                result[key] = [compact(value) for value in result[key]]
        for key in ('items', 'additionalProperties', 'additionalItems', 'contains',
                    'propertyNames', 'not', 'if', 'then', 'else'):
            if isinstance(result.get(key), dict):
                result[key] = compact(result[key])
        # Unknown keywords, dependencies, and literal defaults/enum/const/examples
        # retain their exact values; they are not assumed to be schema nodes.
        return result
    return {name: compact(schema) for name, schema in schemas.items()}


@dataclass(frozen=True)
class ActionResolutionSnapshot:
    """Copied resolver inputs with no reference to an executable session.

    The state and native action objects remain their original model types, but
    are copied when captured. Resolvers treat those copies as read-only and
    return a separate copied action, leaving both the snapshot and session intact.
    """

    current_state: object
    actions_filtered: tuple
    treated_patients: frozenset
    evac_patients: frozenset
    tagged_patients: frozenset
    apply_action_filtering: bool

    @classmethod
    def from_session(cls, session, apply_action_filtering=None):
        """Capture current local data without fetching offers or scenario state."""
        return cls(
            current_state=deepcopy(session.current_state),
            actions_filtered=tuple(deepcopy(session.actions_filtered)),
            treated_patients=frozenset(session.treated_patients),
            evac_patients=frozenset(session.evac_patients),
            tagged_patients=frozenset(session.tagged_patients),
            apply_action_filtering=(
                session.apply_action_filtering if apply_action_filtering is None
                else apply_action_filtering),
        )


def as_dict(obj):
    if isinstance(obj, dict):
        return obj
    return obj.to_dict() if hasattr(obj, "to_dict") else obj._asdict()


def enum_value(value):
    """The plain value behind a swagger enum member (e.g. a supply's
    `type` or a character's `tag`), so that it compares / displays as
    the server's string ('Tourniquet') rather than as
    'SupplyTypeEnum.TOURNIQUET'; non-enum values pass through."""
    return value.value if isinstance(value, Enum) else value


def in_stock_supplies(scenario_state):
    """The scenario state's supplies that are in stock (a supply with
    no reported quantity is assumed available); empty when the state
    doesn't report supplies at all."""
    return [s for s in (getattr(scenario_state, 'supplies', None) or [])
            if s.quantity is None or s.quantity > 0]


def format_available_actions(actions):
    """Describe each available action as `- tool_name: description` so
    the agent can map what the environment offers onto its tools."""
    return "\n".join(
        f"- {TOOL_NAMES_BY_ACTION_TYPE.get(a.action_type, a.action_type)}: "
        f"{a.unstructured}"
        for a in actions)


def untagged_characters(current_state, tagged_patients):
    """Characters still needing a tag.  The (live) environment only
    reports `tag` for nearby characters, so a tag we applied earlier
    disappears from the state once we walk away -- hence the manually
    tracked `tagged_patients` set (as for treated / evac'd patients)."""
    return {c.id for c in current_state.characters
            if c.tag is None and not c.unseen
            and c.id not in tagged_patients}


def filter_available_actions(current_state, available_actions,
                             treated_patients, evac_patients,
                             tagged_patients):
    """Drop actions that no longer make sense for the scenario state:
    tagging when nobody is left untagged, treating / evacuating when
    nobody is left to treat / evacuate (also for actions already
    targeting a specific character).  END_SCENE is not handled here;
    see ScenarioSession.refresh_actions.

    The environment doesn't track which patients have been treated or
    evac'd (via c.unseen or otherwise), and only reports tags for
    nearby characters, hence the manually tracked `treated_patients` /
    `evac_patients` / `tagged_patients` sets."""
    untagged_patients = untagged_characters(
        current_state, tagged_patients)
    treatable_patients = {
        c.id for c in current_state.characters
        if c.id not in treated_patients}

    filtered = []
    for a in available_actions:
        if a.action_type == ActionTypeEnum.TAG_CHARACTER:
            if len(untagged_patients) == 0:
                continue
            if (a.character_id is not None
                    and a.character_id not in untagged_patients):
                continue

        elif a.action_type == ActionTypeEnum.TREAT_PATIENT:
            if len(treatable_patients) == 0:
                continue
            if (a.character_id is not None
                    and a.character_id not in treatable_patients):
                continue

        elif a.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            evacable_patients = {
                c.id for c in current_state.characters
                if c.id not in evac_patients}
            if len(evacable_patients) == 0:
                continue
            if (a.character_id is not None
                    and a.character_id not in evacable_patients):
                continue

        filtered.append(a)

    return filtered


def review_action_view(action, justification=None):
    """Copy only agent-visible action fields, never evaluation labels."""
    fields = ('action_id', 'action_type', 'character_id', 'parameters',
              'unstructured', 'intent_action')
    view = {key: enum_value(getattr(action, key, None)) for key in fields}
    view['justification'] = getattr(action, 'justification', None) or justification
    return json.loads(json.dumps(view, default=str))


def proposal_tool_view(action):
    """Describe a resolved proposal using its native tool's argument names.

    Build from completed parameters, not the original request: resolution
    may have supplied offered defaults or canonicalized inventory names.
    This is descriptive data only and never invokes a tool or refreshes state.
    """
    arguments = {'justification': action.justification}
    if action.action_type in CHARACTER_REQUIRED_ACTIONS:
        # Native action tools accept casualty IDs as well as display names.
        arguments['character_name'] = action.character_id
    parameters = action.parameters or {}
    if action.action_type == ActionTypeEnum.TREAT_PATIENT:
        arguments.update(treatment_supply=parameters['treatment'],
                         injury_location=parameters['location'])
    elif action.action_type == ActionTypeEnum.TAG_CHARACTER:
        arguments['triage_tag'] = parameters['category']
    return deepcopy({
        'tool_name': TOOL_NAMES_BY_ACTION_TYPE[action.action_type],
        'action_id': enum_value(getattr(action, 'action_id', None)),
        'action_type': enum_value(action.action_type),
        'intent_action': getattr(action, 'intent_action', None),
        'arguments': arguments,
    })


def same_action(left, right):
    """Compare executable choices, excluding their explanations."""
    fields = ('action_id', 'action_type', 'character_id',
              'parameters', 'intent_action')
    return all(left[key] == right[key] for key in fields)


def parameter_validation(snapshot, action, checked=True):
    if action.action_type != ActionTypeEnum.TREAT_PATIENT:
        return {}
    return {'inventory': (
        'not_checked' if not checked else
        'unverified' if getattr(snapshot.current_state, 'supplies', None) is None
        else 'verified')}


def match_candidate_for_character(snapshot, tool_name, action_type, candidates,
                                  character_name):
    """Match one of `candidates` to the casualty named by the
    agent, completing a generic (untargeted) candidate if needed.
    Returns (action, None) on success, or (None, error) where
    `error` is the message to send back to the agent."""
    visible_characters = [
        c for c in snapshot.current_state.characters
        if not getattr(c, 'unseen', False)]

    if not character_name:
        names = ", ".join(c.name for c in visible_characters)
        return None, (f"{tool_name} requires a target casualty; call "
                      "it again with character_name set to one of: "
                      f"{names}")

    matched_character = next(
        (c for c in visible_characters
         if character_name.lower() in (c.name.lower(),
                                       c.id.lower())),
        None)

    if matched_character is None:
        names = ", ".join(c.name for c in visible_characters)
        return None, (f"Unknown casualty '{character_name}'; "
                      f"valid casualties are: {names}")

    # Per-casualty guards complementing the action filtering
    # (which can only exclude actions that already name a
    # specific casualty)
    if snapshot.apply_action_filtering:
        if (action_type == ActionTypeEnum.TREAT_PATIENT
                and matched_character.id in snapshot.treated_patients):
            return None, (f"{matched_character.name} has already been "
                          "treated; choose a different casualty or "
                          "action.")
        if (action_type == ActionTypeEnum.MOVE_TO_EVAC
                and matched_character.id in snapshot.evac_patients):
            return None, (f"{matched_character.name} has already been "
                          "moved to evac; choose a different casualty "
                          "or action.")
        if (action_type == ActionTypeEnum.TAG_CHARACTER
                and (matched_character.tag is not None
                     or matched_character.id in snapshot.tagged_patients)):
            return None, (f"{matched_character.name} is already tagged "
                          f"as {enum_value(matched_character.tag)}; "
                          "choose a "
                          "different casualty or action.")

    # Prefer an action already targeting the casualty (if the
    # environment offers per-character actions), otherwise
    # complete a generic (untargeted) one
    targeted_action = next(
        (a for a in candidates
         if a.character_id == matched_character.id),
        None)

    if targeted_action is not None:
        return deepcopy(targeted_action), None

    generic_action = next(
        (a for a in candidates if a.character_id is None),
        None)

    if generic_action is None:
        character_ids_to_names = {
            c.id: c.name for c in visible_characters}
        targets = ", ".join(sorted(
            {character_ids_to_names.get(a.character_id,
                                        a.character_id)
             for a in candidates}))
        return None, (f"{tool_name} is not currently available "
                      f"for {matched_character.name}; it is "
                      f"available for: {targets}")

    action_to_take = deepcopy(generic_action)
    action_to_take.character_id = matched_character.id
    return action_to_take, None


def fill_tag_parameters(action, tool_name, triage_tag):
    """Set the action's triage category from the agent's
    `triage_tag`, in place; the agent's explicit choice overrides
    any category a candidate action already carries.  Returns an
    error message for the agent, or None."""
    if action.parameters is None:
        action.parameters = {}

    if not triage_tag:
        if 'category' in action.parameters:
            return None
        return ("Tagging requires a triage category; call "
                f"{tool_name} again with triage_tag set to "
                f"one of: {', '.join(TRIAGE_TAGS)}")

    matched_tag = next(
        (t for t in TRIAGE_TAGS
         if t.lower() == triage_tag.lower()),
        None)

    if matched_tag is None:
        return (f"Unknown triage_tag '{triage_tag}'; "
                f"valid tags are: {', '.join(TRIAGE_TAGS)}")

    action.parameters['category'] = matched_tag
    return None


def fill_treatment_parameters(snapshot, action, tool_name,
                              treatment_supply, injury_location):
    """Complete parameters without treating unknown inventory as empty.

    Explicit arguments override offered defaults. Validate location even
    when inventory is unknown; inventory availability is checked only when
    supplies are reported. Commit parameters only after validation passes.
    """
    parameters = dict(action.parameters or {})
    treatment = (treatment_supply if treatment_supply
                 else parameters.get('treatment'))
    if not isinstance(treatment, str) or not treatment.strip():
        return (f"Treating requires choosing a supply; call {tool_name} "
                "again with a nonempty treatment_supply.")
    treatment = treatment.strip()
    if treatment.casefold() in TREATMENT_PLACEHOLDERS:
        return (f"Invalid treatment_supply '{treatment}': this is a missing-value "
                f"placeholder, not a treatment name. Call {tool_name} with a "
                "concrete supply name, or choose another available action. "
                "Unknown inventory does not make a placeholder valid.")
    location = (injury_location if injury_location
                else parameters.get('location') or 'unspecified')
    matched_location = next(
        (loc for loc in INJURY_LOCATIONS
         if isinstance(location, str) and loc.lower() == location.strip().lower()),
        None)
    if matched_location is None:
        return (f"Unknown injury_location '{location}'; valid locations are: "
                f"{', '.join(INJURY_LOCATIONS)}")

    if getattr(snapshot.current_state, 'supplies', None) is not None:
        supply_names = [str(enum_value(s.type))
                        for s in in_stock_supplies(snapshot.current_state)]
        matched_supply = next(
            (name for name in supply_names if name.lower() == treatment.lower()),
            None)
        if matched_supply is None:
            return (f"Unknown or out-of-stock treatment_supply '{treatment}'; "
                    f"available supplies are: {', '.join(supply_names) or 'none'}")
        treatment = matched_supply

    parameters.update(treatment=treatment, location=matched_location)
    action.parameters = parameters
    return None


def resolve_typed_action(snapshot, action_type, justification,
                         character_name="", triage_tag="",
                         treatment_supply="", injury_location=""):
    """Resolve a copied action from the current filtered snapshot.

    Return (action, error). Never fetch state, execute tools, or mutate
    offered actions or session bookkeeping, including during review.
    """
    tool_name = TOOL_NAMES_BY_ACTION_TYPE[action_type]
    candidates = [a for a in snapshot.actions_filtered
                  if a.action_type == action_type]

    if not candidates:
        return None, (f"{tool_name} is not currently available.  The "
                      "currently available actions are:\n"
                      + format_available_actions(snapshot.actions_filtered))

    if action_type in CHARACTER_REQUIRED_ACTIONS:
        action_to_take, error = match_candidate_for_character(
            snapshot, tool_name, action_type, candidates,
            character_name)
        if error is not None:
            return None, error
    else:
        if len(candidates) > 1:
            log.info(f"{tool_name}: multiple candidate actions "
                     "offered by the environment; taking the first "
                     f"('{candidates[0].unstructured}')")
        action_to_take = deepcopy(candidates[0])

    if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
        error = fill_tag_parameters(
            action_to_take, tool_name, triage_tag)
        if error is not None:
            return None, error

    if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
        error = fill_treatment_parameters(
            snapshot, action_to_take, tool_name,
            treatment_supply, injury_location)
        if error is not None:
            return None, error

    # The current agent's explanation must replace any offered default.
    action_to_take.justification = justification
    return action_to_take, None
