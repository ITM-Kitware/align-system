import inspect
from inspect import Parameter
import functools
from collections import OrderedDict


def call_with_coerced_args(func, dictionary, partial=False):
    # Create a copy that we can edit and pass along if the func
    # requests **kwargs
    dictionary_copy = dictionary.copy()

    args = []
    kwargs = OrderedDict()
    pass_all = False
    for name, param in inspect.signature(func).parameters.items():
        if name in dictionary:
            if param.kind == Parameter.POSITIONAL_ONLY:
                # Rare case, usually parameters are of kind
                # POSITIONAL_OR_KEYWORD
                args.append(dictionary_copy[name])
                del dictionary_copy[name]
            else:
                kwargs[name] = dictionary_copy[name]
                del dictionary_copy[name]
        elif param.kind == Parameter.VAR_KEYWORD:
            # If func has **kwargs in it's argument signature, just
            # pass everything
            pass_all = True
        elif param.default != inspect._empty:
            pass  # Don't need to add to the arg/kwarg list

    if pass_all:
        kwargs = {**kwargs, **dictionary_copy}

    if partial:
        return functools.partial(func, *args, **kwargs)
    else:
        return func(*args, **kwargs)
