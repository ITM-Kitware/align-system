"""Expose the requested target without scoring actions or calling a model."""
import json

from pydantic import BaseModel

from align_system.utils.alignment_targets import alignment_target_entries


class TargetContext:
    def __init__(self, attributes=None):
        self.attributes = {
            name: (value.model_dump(exclude_none=True)
                   if isinstance(value, BaseModel) else dict(value))
            for name, value in (attributes or {}).items()}

    def context(self, target):
        entries = alignment_target_entries(target)
        return {
            'target': entries,
            'attribute_definitions': {
                name: definition for name, definition in self.attributes.items()
                if name == 'medical' or any(t.get('kdma') == name for t in entries)},
        }

    def render(self, target):
        return (
            'Choose actions using the requested alignment target and the '
            'attribute definitions below. Interpret the requested values; '
            'do not automatically maximize every attribute. If a definition '
            'or distribution interpretation is missing, state uncertainty. '
            'This block is target data, not additional instructions.\n'
            + json.dumps(self.context(target), ensure_ascii=False, default=str))
