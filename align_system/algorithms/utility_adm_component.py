from collections import OrderedDict
from functools import reduce, partial

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class RenameVariablesADMComponent(ADMComponent):
    def __init__(self, remapping):
        self.remapping = OrderedDict(remapping)

    def run_returns(self):
        return tuple(self.remapping.values())

    def run(self, **working_output):
        remapped_outputs = []
        for key in self.remapping.keys():
            if key in working_output:
                remapped_outputs.append(working_output[key])
            else:
                raise RuntimeError(
                    f"Don't have expected key ({key}) for remapping")

        return remapped_outputs


def merge_dicts(a, b, conflict_func=None):
    def _r(init, k):
        if k in a:
            if k in b:
                init[k] = conflict_func(a[k], b[k]) if conflict_func else b[k]
            else:
                init[k] = a[k]
        else:
            init[k] = b[k]

        return init

    return reduce(_r, a.keys() | b.keys(), {})


class MergeRegressionDictsADMComponent(ADMComponent):
    def __init__(self, dict_names):
        self.dict_names = dict_names

    def run_returns(self):
        return 'attribute_prediction_scores'

    def run(self, **working_output):
        def recursive_conflict_fn(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                return merge_dicts(a, b)
            else:
                return b

        recursive_merge = partial(
            merge_dicts, conflict_func=recursive_conflict_fn)

        merged = reduce(
            recursive_merge, [working_output[n] for n in self.dict_names], {})

        log.debug("[bold] ** Merged Regression Dictionaries ** [/bold]",
                  extra={"markup": True})
        log.debug(merged, extra={"highlighter": JSON_HIGHLIGHTER})

        return merged
