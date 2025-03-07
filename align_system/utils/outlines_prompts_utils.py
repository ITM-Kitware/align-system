import yaml
import json
import outlines
import copy
from functools import reduce


def remove_identical_attributes(dicts):
    '''
    Removes common attributes across dicts (including nested items)
    '''
    from collections import defaultdict

    def normalize_dicts(dicts):
        '''
        Normalization for detecting equivalent attributes
        For now just includes that left injury = right injury
        TODO - Could bin age groups so age is not included if difference is small
        '''
        norm_dicts = copy.deepcopy(dicts)
        for d in norm_dicts:
            if 'injuries' in d:
                for injury in d['injuries']:
                    if 'location' in injury:
                        injury['location'] = injury['location'].replace('left ','').replace('right ', '')
        return norm_dicts

    def get_common_values(dicts):
        """Recursively find common values across dictionaries."""
        common = defaultdict(list)
        all_keys = set()

        norm_dicts = normalize_dicts(dicts)

        # Collect all keys from all dictionaries
        for d in dicts:
            all_keys.update(d.keys())

        for key in all_keys:
            values = [d.get(key) for d in norm_dicts]
            if isinstance(values[0], dict):
                # If the value is a dictionary, handle it recursively
                nested_common = get_common_values([v for v in values if isinstance(v, dict)])
                if nested_common:
                    common[key] = nested_common
            elif isinstance(values[0], list) and all(isinstance(v, dict) for v in values):
                # If the value is a list of dictionaries, handle it recursively
                list_common = get_common_values([item for sublist in values for item in sublist])
                if list_common:
                    common[key] = list_common
            else:
                # Handle non-dictionary, non-list values
                if all(value == values[0] for value in values):
                    common[key] = values[0]

        return common

    def filter_dict(d, common):
        """Filter out common attributes from the dictionary."""
        if not isinstance(d, dict):
            return d

        filtered = {}
        for k, v in d.items():
            if k in common:
                if isinstance(v, dict):
                    nested_filtered = filter_dict(v, common[k])
                    if nested_filtered:
                        filtered[k] = nested_filtered
                elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    # Don't include common injuries
                    pass
                else:
                    # Include only if the attribute value is not common
                    if v != common[k]:
                        filtered[k] = v
            else:
                filtered[k] = v

        return filtered

    # Get common attributes
    common_values = get_common_values(dicts)

    # Process each dictionary to remove common attributes
    return [filter_dict(d, common_values) for d in dicts]


def get_unique_structured_character_info(characters):
    '''
    Returns a list of character dicts with: name, unstrucutured, id, and relevant_structured
    # where relevant_structured is a string of info unique to each character
    '''
    character_dicts = []
    for character in characters:
        character_dicts.append(character.to_dict()) # convert to dict

    # Remove info that is the same across character_dicts
    relevant_structured_dicts = remove_identical_attributes(character_dicts)

    # Remove unstructured info from structured dicts
    for relevant_structured_dict in relevant_structured_dicts:
        relevant_structured_dict.pop('name', None)
        relevant_structured_dict.pop('id', None)
        relevant_structured_dict.pop('unstructured', None)

    return_character_dicts = []
    for i in range(len(character_dicts)):
        return_character_dict = {}
        return_character_dict['name'] = character_dicts[i]['name']
        return_character_dict['id'] = character_dicts[i]['id']
        return_character_dict['unstructured'] = character_dicts[i]['unstructured']
        # Serialize as strings for prompt
        return_character_dict['relevant_structured'] = json.dumps(relevant_structured_dicts[i])
        return_character_dicts.append(return_character_dict)

    return return_character_dicts

def remove_null_values(data):
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_null_values(d) for d in data if isinstance(d, dict)]
    return data

def get_relevant_structured_character_info(characters, target_kdmas):
    '''
    Returns a list of character dicts with: name, unstrucutured, id, and relevant_structured
    where relevant_structured is a string of relevant strucutured character info
    '''
    relevant_fields = []
    for target_kdma in target_kdmas:
        relevant_fields.extend(target_kdma['relevant_structured_character_info'])
    return_character_dicts = []
    for character in characters:
        character_dict = character.to_dict()
        return_character_dict = {}
        return_character_dict['name'] = character_dict['name']
        return_character_dict['id'] = character_dict['id']
        return_character_dict['unstructured'] = character_dict['unstructured']
        relevant_structured_dict = {}
        for field in relevant_fields:
            value = reduce(lambda d, key: d.get(key) if d else None, field.split('.'), character_dict)
            relevant_structured_dict[field] = remove_null_values(value)
        if relevant_structured_dict:
            return_character_dict['relevant_structured'] = json.dumps(relevant_structured_dict)
        else:
            return_character_dict['relevant_structured'] = None
        return_character_dicts.append(return_character_dict)

    return return_character_dicts

def new_get_relevant_structured_character_info(characters, relevant_fields):
    '''
    Returns a list of character dicts with: name, unstrucutured, id, and relevant_structured
    where relevant_structured is a string of relevant strucutured character info
    '''
    return_character_dicts = []
    for character in characters:
        character_dict = character.to_dict()
        return_character_dict = {}
        return_character_dict['name'] = character_dict['name']
        return_character_dict['id'] = character_dict['id']
        return_character_dict['unstructured'] = character_dict['unstructured']
        relevant_structured_dict = {}
        for field in relevant_fields:
            value = reduce(lambda d, key: d.get(key) if d else None, field.split('.'), character_dict)
            relevant_structured_dict[field] = remove_null_values(value)
        if relevant_structured_dict:
            return_character_dict['relevant_structured'] = json.dumps(relevant_structured_dict)
        else:
            return_character_dict['relevant_structured'] = None
        return_character_dicts.append(return_character_dict)

    return return_character_dicts
