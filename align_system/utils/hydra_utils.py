from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from hydra.utils import instantiate


def initialize_with_custom_references(cfg):
    # Need to 'allow_objects' to allow arbitrary objects to be stored
    # in the OmegaConf config
    cfg = OmegaConf.create(cfg, flags={"allow_objects": True})

    # Need to do an interpolation first pass where non-"ref"
    # interpolations are resolved.  This eliminates a race condition
    # where depending on the config order, normal reference
    # (i.e. "{path}") coming after "ref" references could point to the
    # same object.
    OmegaConf.register_new_resolver('ref', lambda path: f'${{ref:{path}}}')
    OmegaConf.resolve(cfg)
    # Remove our custom resolver
    OmegaConf.clear_resolver('ref')

    # Creating a custom resolver here to ensure that when our custom
    # "ref" (${ref:<path_to_referenced_object>}) variable in the
    # config is accessed, that the thing it points to is instantiated
    # and then a reference to that instantiated object is returned.
    #
    # If we naively try to use the standard
    # ${<path_to_referenced_object>} variable interpolation we'll end
    # up with multiple instances of the same object (one instance for
    # the original, and another for each variable reference in the
    # config).  This is a real problem when our objects which might
    # take up a tremendous amount of resources (e.g. LLMs)
    def _custom_resolver(path, *, _root_):
        node = OmegaConf.select(_root_, path)

        if node is None:
            raise InterpolationResolutionError(
                f"Couldn't find referenced node '${{ref:{path}}}'")

        *prefixes, base = path.split('.')

        prefix = '.'.join(prefixes)
        parent = OmegaConf.select(_root_, prefix)

        # Replace with instantiated
        if isinstance(node, DictConfig) or isinstance(node, ListConfig):
            instantiated_node = instantiate(node, _recursive_=True)
            parent[base] = instantiated_node
        else:
            # Assume already instantiated
            instantiated_node = node

        return instantiated_node

    OmegaConf.register_new_resolver('ref', _custom_resolver)

    # Finally perform the recursive instantiation (via Hydra), now as
    # Hydra hits our custom references our custom resolver should
    # instantiate the original object first and point to it rather
    # than make a new copy
    cfg = instantiate(cfg, recursive=True)

    # Remove our custom resolver
    OmegaConf.clear_resolver('ref')

    return cfg
