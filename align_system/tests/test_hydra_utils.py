from align_system.utils.hydra_utils import initialize_with_custom_references


class Stub():
    pass


def test_custom_ref_instantiate_1():
    input_cfg = {'a': {'_target_': 'align_system.tests.test_hydra_utils.Stub'},
                 'b': '${ref:a}'}

    output_cfg = initialize_with_custom_references(input_cfg)

    assert(id(output_cfg.a) == id(output_cfg.b))


def test_custom_ref_instantiate_2():
    input_cfg = {'a': {'_target_': 'align_system.tests.test_hydra_utils.Stub'},
                 'b': '${ref:a}',
                 'c': '${a}'}

    output_cfg = initialize_with_custom_references(input_cfg)

    assert(id(output_cfg.a) == id(output_cfg.b))
    assert(id(output_cfg.a) != id(output_cfg.c))


def test_custom_ref_instantiate_3():
    input_cfg = {'a': {'_target_': 'align_system.tests.test_hydra_utils.Stub'},
                 'c': '${a}',
                 'b': '${ref:a}'}

    output_cfg = initialize_with_custom_references(input_cfg)

    assert(id(output_cfg.a) == id(output_cfg.b))
    assert(id(output_cfg.a) != id(output_cfg.c))
