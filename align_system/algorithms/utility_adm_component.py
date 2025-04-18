from collections import OrderedDict

from align_system.algorithms.abstracts import ADMComponent


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
