import pytest
from contextlib import nullcontext as does_not_raise

from align_system.algorithms.alignment_adm_component import MedicalUrgencyAlignmentADMComponent


class TestMedicalUrgencyAlignmentADMComponent:
    attribute_definitions = {
        "KDMA_A": {
            "name": "KDMA A",
            "kdma": "KDMA_A",
            "description": "Test KDMA A",
        },
        "KDMA_B": {
            "name": "KDMA B",
            "kdma": "KDMA_B",
            "description": "Test KDMA B",
        }
    }

    @pytest.mark.parametrize(
        ("attribute_prediction_scores", "alignment_target", "exp_choice", "exp_raises"),
        [
            # No alignment target
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"medical": 0.6, "KDMA_A": 0.3, "KDMA_B": 0.5},
                },
                None,
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"Assumption violated: `alignment_target` was None"),
            ),
            # Multi-kdma alignment
            (
                {
                    "Choice 0": {"medical": 0.2, "KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"medical": 0.5, "KDMA_A": 0.3, "KDMA_B": 0.5},
                },
                {
                    "kdma_values":
                    [
                        {"kdma": "KDMA_A", "value": 0.3},
                        {"kdma": "KDMA_B", "value": 0.7},
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(NotImplementedError, match=r"Multi-kdma alignment"),
            ),
            # >2 choices
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.2, "KDMA_A": 0.3},
                    "Choice 2": {"medical": 0.6, "KDMA_A": 0.4},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.3}],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(NotImplementedError, match=r"This alignment function has not yet been"),
            ),
            # <2 choices
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_B": 0.8},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_B", "value": 0.7}],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(NotImplementedError, match=r"This alignment function has not yet been"),
            ),
            # Same medical
            (
                {
                    "Choice 0": {"medical": 0.5, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.5, "KDMA_A": 0.9},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.3}],
                },
                "Choice 1",  # Attribute worthy patient
                does_not_raise(),
            ),
            # Same patient is medically AND attribute worthy
            (
                {
                    "Choice 0": {"medical": 0.5, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.9, "KDMA_A": 0.9},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.3}],
                },
                "Choice 1",
                does_not_raise(),
            ),
            # Target above midpoint (0.35)
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.7}],
                },
                "Choice 1",  # Choose attribute-worthy patient
                does_not_raise(),
            ),
            # Target below midpoint (0.35)
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.1}],
                },
                "Choice 0",  # Chose medically-worthy patient
                does_not_raise(),
            ),
            # Target above midpoint (0.55)
            (
                {
                    "Choice 0": {"medical": 0.3, "KDMA_B": 0.8},
                    "Choice 1": {"medical": 0.7, "KDMA_B": 0.5},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_B", "value": 0.65}],
                },
                "Choice 0",  # Choose attribute-worthy patient
                does_not_raise(),
            ),
            # Target below midpoint (0.55)
            # Extra KDMAs should be ignored
            (
                {
                    "Choice 0": {"medical": 0.3, "KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.9, "KDMA_B": 0.5},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_B", "value": 0.25}],
                },
                "Choice 1",  # Choose medically-worthy patient
                does_not_raise(),
            ),
            # Target above midpoint (0.8)
            (
                {
                    "Choice 0": {"medical": 0.2, "KDMA_A": 0.6},
                    "Choice 1": {"medical": 0.9, "KDMA_A": 0.5},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.9}],
                },
                "Choice 0",  # Choose attribute-worthy patient
                does_not_raise(),
            ),
            # Target below midpoint (0.8)
            (
                {
                    "Choice 0": {"medical": 0.2, "KDMA_A": 0.6},
                    "Choice 1": {"medical": 0.9, "KDMA_A": 0.5},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.3}],
                },
                "Choice 1",  # Chose medically-worthy patient
                does_not_raise(),
            ),
            # Multiple predictions, predictions of different lengths, target above midpoint (0.625)
            (
                {
                    "Choice 0": {"medical": [0.2, 0.3, 0.3, 0.2], "KDMA_A": [0.6, 0.8]},  # medical: 0.25, KDMA_A: 0.7
                    "Choice 1": {"medical": [0.9, 0.5, 0.9, 0.5], "KDMA_A": [0.5]},  # medical: 0.7, KDMA_A: 0.5
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.7}],
                },
                "Choice 0",  # Chose attribute-worthy patient
                does_not_raise(),
            ),
            # Multiple predictions, predictions of different lengths, target below midpoint (0.625)
            (
                {
                    "Choice 0": {"medical": [0.2, 0.3, 0.3, 0.2], "KDMA_A": [0.6, 0.8]},  # medical: 0.25, KDMA_A: 0.7
                    "Choice 1": {"medical": [0.9, 0.5, 0.9, 0.5], "KDMA_A": [0.5]},  # medical: 0.7, KDMA_A: 0.5
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.6}],
                },
                "Choice 1",  # Chose medically-worthy patient
                does_not_raise(),
            ),
            # We choose randomly for target == midpoint, so no guarantees there
        ],
        ids=[
            "no target", "multi-kdma", ">2 choices", "<2 choices", "same medical",
            "same medical and attribute patient", "target above midpoint (0.35)", "target below midpoint (0.35)",
            "target above midpoint (0.55)", "target below midpoint (0.55), extraneous KDMAs",
            "target above midpoint (0.8)", "target below midpoint (0.8)", "multiple predictions, target above midpoint (0.625)",
            "multiple predictions, target below midpoint (0.625)"
        ],
    )
    def test_run(self, attribute_prediction_scores, alignment_target, exp_choice, exp_raises):
        """ Test expected outcomes """
        alignment_fn = MedicalUrgencyAlignmentADMComponent(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        with exp_raises:
            # Only checking selected choice as best sample index not yet implemented
            assert alignment_fn.run(attribute_prediction_scores, alignment_target)[0] == exp_choice

    @pytest.mark.parametrize(
        ("medical_delta", "attribute_delta", "exp_value"),
        [
            (0.7, 0.1, 0.8),
            (0.3, 0.2, 0.55),
            (0.9, 0.4, 0.75),
            (0.1, 0.8, 0.15),
        ],
    )
    def test_midpoint_eqn(self, medical_delta, attribute_delta, exp_value):
        """ Regression test to ensure equation doesn't get inadvertently modified """
        alignment_fn = MedicalUrgencyAlignmentADMComponent(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        assert alignment_fn._midpoint_eqn(medical_delta, attribute_delta) == pytest.approx(exp_value)
