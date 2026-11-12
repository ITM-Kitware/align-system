"""Normalize alignment targets without loading alignment algorithms."""
from collections.abc import Mapping


def alignment_target_entries(alignment_target):
    """Return target entries as dictionaries, preserving non-scalar fields."""
    if alignment_target is None:
        return []
    entries = (alignment_target.get('kdma_values')
               if isinstance(alignment_target, Mapping)
               else getattr(alignment_target, 'kdma_values', None))
    result = []
    for entry in entries or []:
        if hasattr(entry, 'to_dict'):
            entry = entry.to_dict()
        result.append(dict(entry) if isinstance(entry, Mapping) else vars(entry).copy())
    return result


def alignment_target_values(alignment_target):
    """Return scalar targets; distribution-only entries have no scalar value."""
    return {entry['kdma']: float(entry['value'])
            for entry in alignment_target_entries(alignment_target)
            if entry.get('kdma') is not None and entry.get('value') is not None}
