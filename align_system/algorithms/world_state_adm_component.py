from copy import deepcopy

from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import logging

log = logging.getLogger(__name__)


def _get_field(obj, field, default=None):
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _set_field(obj, field, value):
    if isinstance(obj, dict):
        obj[field] = value
    else:
        setattr(obj, field, value)


def _get_items(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    elif hasattr(obj, 'to_dict'):
        return list(obj.to_dict().items())
    elif hasattr(obj, '__dict__'):
        return list(obj.__dict__.items())
    return []


def _merge_dict_or_obj(existing_obj, new_obj, preserve_if_new_none=False):
    """
    Recursively merges new_obj into existing_obj.
    Supports both dicts and objects with properties/attributes.
    """
    if new_obj is None:
        return existing_obj

    if existing_obj is None:
        return new_obj

    for key, new_val in _get_items(new_obj):
        if new_val is None and preserve_if_new_none:
            continue

        existing_val = _get_field(existing_obj, key)

        if (isinstance(new_val, dict) or hasattr(new_val, 'to_dict')) and not isinstance(new_val, list):
            if existing_val is not None:
                merged_sub = _merge_dict_or_obj(existing_val, new_val, preserve_if_new_none=preserve_if_new_none)
                _set_field(existing_obj, key, merged_sub)
            else:
                _set_field(existing_obj, key, new_val)
        else:
            _set_field(existing_obj, key, new_val)

    return existing_obj


class WorldStateTrackerADMComponent(ADMComponent):
    """
    ADM Component that maintains cumulative world state across scenes/probes in a scenario.

    Handles recursive merging of character details (e.g. `vitals`). Preserves detailed character
    information acquired when `nearby == True` and vitals assessed during interaction, preventing
    less-detailed descriptions from overwriting them when moving away (`nearby == False`). Dynamic
    attributes (e.g. `tag`, `visited`, `unseen`, top-level `supplies`) continue to update normally.
    """
    def __init__(self,
                 output_scenario_state_key='scenario_state',
                 output_original_state_key='original_scenario_state',
                 output_stale_info_key='world_state_stale_info'):
        self.output_scenario_state_key = output_scenario_state_key
        self.output_original_state_key = output_original_state_key
        self.output_stale_info_key = output_stale_info_key

        self.current_scenario_id = None
        self.world_state = None
        self.stale_characters = {}

    def reset_history(self):
        log.info("[bold]*Resetting WorldStateTrackerADMComponent history*[/bold]",
                 extra={"markup": True})
        self.current_scenario_id = None
        self.world_state = None
        self.stale_characters = {}

    def run_returns(self):
        return (self.output_scenario_state_key,
                self.output_original_state_key,
                self.output_stale_info_key)

    def run(self, scenario_state, scenario_id=None):
        # Auto-reset if scenario_id changes unexpectedly without an explicit reset_history call
        if scenario_id is not None and scenario_id != self.current_scenario_id:
            if self.current_scenario_id is not None:
                log.info(f"Scenario ID changed from {self.current_scenario_id} to {scenario_id}, resetting world state")
                self.reset_history()
            self.current_scenario_id = scenario_id

        original_scenario_state = scenario_state
        # Deepcopy the incoming scenario_state up front to ensure complete reference independence
        new_state = deepcopy(scenario_state)

        if self.world_state is None:
            # First scene of the scenario: initialize world state as deep copy of incoming state
            self.world_state = new_state
            self.stale_characters = {}
        else:
            # Merge incoming state into cumulative world_state
            self._merge_scenario_state(new_state)

        stale_info = {
            'stale_characters': deepcopy(self.stale_characters)
        }

        return self.world_state, original_scenario_state, stale_info

    def _merge_scenario_state(self, new_state):
        # 1. Update top-level fields (unstructured narrative, elapsed_time, meta_info, supplies, etc.)
        for key, val in _get_items(new_state):
            if key != 'characters':
                _set_field(self.world_state, key, val)

        # 2. Merge characters
        existing_chars = _get_field(self.world_state, 'characters')
        if existing_chars is None:
            existing_chars = []
            _set_field(self.world_state, 'characters', existing_chars)

        existing_char_map = {}
        for char in existing_chars:
            cid = _get_field(char, 'id') or _get_field(char, 'name')
            if cid:
                existing_char_map[cid] = char

        new_chars = _get_field(new_state, 'characters') or []

        for new_char in new_chars:
            cid = _get_field(new_char, 'id') or _get_field(new_char, 'name')
            if not cid:
                existing_chars.append(new_char)
                continue

            if cid not in existing_char_map:
                existing_chars.append(new_char)
                existing_char_map[cid] = new_char
            else:
                existing_char = existing_char_map[cid]
                self._merge_character(existing_char, new_char, cid)

    def _merge_character(self, existing_char, new_char, cid):
        old_nearby = bool(_get_field(existing_char, 'nearby', False))
        new_nearby = bool(_get_field(new_char, 'nearby', False))
        stale_fields_for_char = []

        if old_nearby and not new_nearby:
            # Character was previously nearby (rich detail) but is now not nearby
            old_unstructured = _get_field(existing_char, 'unstructured')
            old_vitals = deepcopy(_get_field(existing_char, 'vitals'))

            # Update all fields from new_char
            for key, val in _get_items(new_char):
                if key not in {'unstructured', 'vitals'}:
                    _set_field(existing_char, key, val)

            # Preserve richer unstructured description
            if old_unstructured is not None:
                _set_field(existing_char, 'unstructured', old_unstructured)
                stale_fields_for_char.append('unstructured')

            # Merge/preserve vitals non-destructively
            new_vitals = _get_field(new_char, 'vitals')
            if old_vitals is not None:
                if new_vitals is None:
                    _set_field(existing_char, 'vitals', old_vitals)
                else:
                    merged_vitals = _merge_dict_or_obj(old_vitals, new_vitals, preserve_if_new_none=True)
                    _set_field(existing_char, 'vitals', merged_vitals)
                stale_fields_for_char.append('vitals')
            elif new_vitals is not None:
                _set_field(existing_char, 'vitals', new_vitals)

            if stale_fields_for_char:
                self.stale_characters[cid] = {
                    'stale_fields': stale_fields_for_char,
                    'nearby': False
                }
                log.debug(f"Preserving detailed info ({stale_fields_for_char}) for non-nearby character '{cid}'")
        else:
            # Character is now nearby (or was never nearby) -> recursive update
            old_vitals = _get_field(existing_char, 'vitals')

            for key, val in _get_items(new_char):
                if key == 'vitals':
                    if old_vitals is not None and val is not None:
                        merged_vitals = _merge_dict_or_obj(old_vitals, val)
                        _set_field(existing_char, key, merged_vitals)
                    elif val is not None:
                        _set_field(existing_char, key, val)
                else:
                    _set_field(existing_char, key, val)

            if cid in self.stale_characters:
                del self.stale_characters[cid]
