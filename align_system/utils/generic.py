import inspect
from inspect import Parameter
import functools


def call_with_coerced_args(func, dictionary, partial=False):
    args = []
    kwargs = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in dictionary:
            if param.kind == Parameter.POSITIONAL_ONLY:
                # Rare case, usually parameters are of kind
                # POSITIONAL_OR_KEYWORD
                args.append(dictionary[name])
            else:
                kwargs[name] = dictionary[name]
        elif param.default != inspect._empty:
            pass  # Don't need to add to the arg/kwarg list
        else:
            if not partial:
                raise RuntimeError(f"Don't have expected parameter "
                                   f"('{name}') in provided dictionary")

    if partial:
        return functools.partial(func, *args, **kwargs)
    else:
        return func(*args, **kwargs)
