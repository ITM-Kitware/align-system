import pytest
import unittest.mock as mock
from contextlib import nullcontext as does_not_raise

from align_system.algorithms.alignment_adm_component import (
    MedicalUrgencyAlignmentADMComponent,
    MedicalUrgencyAlignmentWeightedADMComponent,
    RandomEffectsModelAlignmentADMComponent)

@pytest.mark.parametrize(
    ("alignment_fn_class"),
    [
        MedicalUrgencyAlignmentADMComponent,
        MedicalUrgencyAlignmentWeightedADMComponent,
    ]
 )
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
        },
        "KDMA_C": {
            "name": "KDMA C",
            "kdma": "KDMA_C",
            "description": "Test KDMA C",
        },
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
            # No medical predictions
            (
                {
                    "Choice 0": {"KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"KDMA_A": 0.3, "KDMA_B": 0.5},
                },
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.3}],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"Medical Urgency predictions required"),
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
            "no target", "no medical preds", ">2 choices", "<2 choices", "same medical",
            "same medical and attribute patient", "target above midpoint (0.35)", "target below midpoint (0.35)",
            "target above midpoint (0.55)", "target below midpoint (0.55), extraneous KDMAs",
            "target above midpoint (0.8)", "target below midpoint (0.8)", "multiple predictions, target above midpoint (0.625)",
            "multiple predictions, target below midpoint (0.625)"
        ],
    )
    def test_run(self, alignment_fn_class, attribute_prediction_scores, alignment_target, exp_choice, exp_raises):
        """ Test expected outcomes """
        alignment_fn = alignment_fn_class(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        with exp_raises:
            # Only checking selected choice as best sample index not yet implemented
            assert alignment_fn.run(attribute_prediction_scores, alignment_target)[0] == exp_choice

    @pytest.mark.parametrize(
        ("attribute_prediction_scores", "alignment_target", "exp_choice"),
        [
            # Same medical, one patient favored by all attributes
            (
                {
                    "Choice 0": {"medical": 0.5, "KDMA_A": 0.1, "KDMA_B": 0.2},
                    "Choice 1": {"medical": 0.5, "KDMA_A": 0.9, "KDMA_B": 0.7},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.3},
                        {"kdma": "KDMA_B", "value": 0.7},
                    ],
                },
                "Choice 1",  # Attribute worthy patient
            ),
            # Same medical, one attribute tied
            (
                {
                    "Choice 0": {"medical": 0.5, "KDMA_A": 0.9, "KDMA_B": 0.4},
                    "Choice 1": {"medical": 0.5, "KDMA_A": 0.9, "KDMA_B": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.1},
                    ],
                },
                "Choice 0",  # Attribute worthy patient
            ),
            # Fully tied patients chooses the "first" choice for determinism,
            (
                {
                    "Choice 0": {"medical": 0.5, "KDMA_A": 0.9, "KDMA_B": 0.4},
                    "Choice 1": {"medical": 0.5, "KDMA_A": 0.9, "KDMA_B": 0.4},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.1},
                    ],
                },
                "Choice 0",  # First patient
            ),
            # Same patient is medically and attribute favored
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.2},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.9, "KDMA_B": 0.7},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.6},
                        {"kdma": "KDMA_B", "value": 0.1},
                    ],
                },
                "Choice 1",  # Medically and attribute worthy patient
            ),
            # Same medical/attr favored patient for KDMA_A, KDMA_B midpoint is 0.55
            # Targets above 0.55 for KDMA_B would be tie vote
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.9, "KDMA_B": 0.2}, # KDMA_A, KDMA_B if target below 0.55
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.1},
                        {"kdma": "KDMA_B", "value": 0.3},
                    ],
                },
                "Choice 1",
            ),
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.9, "KDMA_B": 0.2}, # KDMA_A, KDMA_B if target below 0.55
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.1},
                        {"kdma": "KDMA_B", "value": 0.9},
                    ],
                },
                "Choice 1",  # Tie vote, choose first patient for determinism (after sorting descending medically)
            ),
            # Same as previous but new target for KDMA_A (shouldn't matter)
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.9, "KDMA_B": 0.2}, # KDMA_A, KDMA_B if target below 0.55
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.3},
                    ],
                },
                "Choice 1",
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.1},
                        {"kdma": "KDMA_B", "value": 0.3},
                    ],
                },
                "Choice 1",  # Both targets below midpoint
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                "Choice 0",  # Both targets above midpoint
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.75},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                "Choice 1",  # KDMA_A target is exactly midpoint so it votes for medically needy
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55. KDMA_C isn't in target so it should be ignored
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.75},
                        {"kdma": "KDMA_B", "value": 0.2},
                    ],
                },
                "Choice 1",  # KDMA_A target is exactly midpoint so it votes for medically needy
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.6},
                        {"kdma": "KDMA_C", "value": 0.8},
                    ],
                },
                "Choice 0",  # All targets above midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.6},
                        {"kdma": "KDMA_C", "value": 0.2},
                    ],
                },
                "Choice 0",  # 2/3 above midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.1},
                        {"kdma": "KDMA_C", "value": 0.8},
                    ],
                },
                "Choice 0",  # 2/3 above midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.5},
                        {"kdma": "KDMA_B", "value": 0.6},
                        {"kdma": "KDMA_C", "value": 0.8},
                    ],
                },
                "Choice 0",  # 2/3 above midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.5},
                        {"kdma": "KDMA_B", "value": 0.2},
                        {"kdma": "KDMA_C", "value": 0.8},
                    ],
                },
                "Choice 1",  # 2/3 below midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.5},
                        {"kdma": "KDMA_B", "value": 0.2},
                        {"kdma": "KDMA_C", "value": 0.4},
                    ],
                },
                "Choice 1",  # 2/3 below midpoint, other exactly midpoint
            ),
            # More than 2 targets. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55, KDMA_C midpoint is 0.4
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7, "KDMA_C": 0.9},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2, "KDMA_C": 0.1},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.75},
                        {"kdma": "KDMA_B", "value": 0.2},
                        {"kdma": "KDMA_C", "value": 0.4},
                    ],
                },
                "Choice 1",  # 1 below midpoint, other 2 exactly midpoint
            ),
            # Multiple predictions. KDMA_A midpoint is 0.5, KDMA_B midpoint is 0.325
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": [0.6, 0.5, 0.4], "KDMA_B": [0.7, 0.6]},  # KDMA_A: 0.5, KDMA_B: 0.65
                    "Choice 1": {"medical": [0.7, 0.6], "KDMA_A": [0.5, 0.3], "KDMA_B": 0.2},  # medical: 0.6, KDMA_A: 0.4
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.75},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                "Choice 0",  # Both targets above midpoint
            ),
        ],
    )
    def test_run_with_multi_kdma(
        self, alignment_fn_class, attribute_prediction_scores, alignment_target, exp_choice
    ):
        """ Test expected outcomes """
        alignment_fn = alignment_fn_class(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        # Only checking selected choice as best sample index not yet implemented
        assert alignment_fn.run(attribute_prediction_scores, alignment_target)[0] == exp_choice

    @pytest.mark.parametrize(
        ("attribute_prediction_scores", "alignment_target", "attribute_relevance", "exp_choice"),
        [
            # Binary relevance. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.2},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                {"KDMA_A": 1.0, "KDMA_B": 0.0},
                "Choice 1",  # Target below midpoint
            ),
            # Same as previous, change KDMA B target (shouldn't matter)
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.2},
                        {"kdma": "KDMA_B", "value": 0.2},
                    ],
                },
                {"KDMA_A": 1.0, "KDMA_B": 0.0},
                "Choice 1",  # Target below midpoint
            ),
            # Binary relevance. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                {"KDMA_A": 1.0, "KDMA_B": 0.0},
                "Choice 0",  # Target above midpoint
            ),
            # Binary relevance. KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.9},
                        {"kdma": "KDMA_B", "value": 0.6},
                    ],
                },
                {"KDMA_A": 0.0, "KDMA_B": 1.0},
                "Choice 0",  # Target above midpoint
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 1.0},
                        {"kdma": "KDMA_B", "value": 0.75},
                    ],
                },
                {"KDMA_A": 0.25, "KDMA_B": 0.5},
                "Choice 0",  # 0.25 to Choice 0, 0.5 to Choice 0 -> Choice 0
            ),
            # KDMA_A midpoint is 0.75, KDMA_B midpoint is 0.55
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.6, "KDMA_B": 0.7},
                    "Choice 1": {"medical": 0.7, "KDMA_A": 0.5, "KDMA_B": 0.2},
                },
                {
                    "kdma_values": [
                        {"kdma": "KDMA_A", "value": 0.6},
                        {"kdma": "KDMA_B", "value": 0.7},
                    ],
                },
                {"KDMA_A": 0.25, "KDMA_B": 0.5},
                "Choice 0",  # 0.25 to Choice 1, 0.5 to Choice 0 -> Choice 0
            ),
        ],
    )
    def test_run_with_explicit_relevance(
        self, alignment_fn_class, attribute_prediction_scores, alignment_target, attribute_relevance, exp_choice
    ):
        """ Test expected outcomes """
        alignment_fn = alignment_fn_class(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        # Only checking selected choice as best sample index not yet implemented
        assert alignment_fn.run(attribute_prediction_scores, alignment_target, attribute_relevance)[0] == exp_choice

    @pytest.mark.parametrize(
        ("kdma", "opt_a_value", "medical_delta", "attribute_delta", "exp_value"),
        [
            ("KDMA_A", 0.3, 0.7, 0.1, 0.8),
            ("KDMA_A", 0.5, 0.3, 0.2, 0.55),
            ("KDMA_A", 0.6, 0.9, 0.4, 0.75),
            ("KDMA_A", 0.0, 0.1, 0.8, 0.15),
            ("KDMA_A", 0.7, 0.7, 0.1, 0.8),  # Should be the exact same as the otherwise equivalent opt_a case
            ("KDMA_A", 1.0, 0.3, 0.2, 0.55),
            ("KDMA_A", 0.65, 0.1, 0.8, 0.15),
        ],
    )
    def test_midpoint_eqn(self, alignment_fn_class, kdma, opt_a_value, medical_delta, attribute_delta, exp_value):
        """ Regression test to ensure equation doesn't get inadvertently modified """
        alignment_fn = alignment_fn_class(
            TestMedicalUrgencyAlignmentADMComponent.attribute_definitions
        )

        assert (
            alignment_fn._midpoint_eqn(kdma, opt_a_value, medical_delta, attribute_delta) == pytest.approx(exp_value)
        )


class TestMedicalUrgencyAlignmentWeightedADMComponent:
    attribute_definitions = {
        "KDMA_A": {
            "name": "Merit Focus",
            "kdma": "merit",
            "description": "Test merit focus KDMA",
        },
        "KDMA_B": {
            "name": "Affiliation Focus",
            "kdma": "affiliation",
            "description": "Test affiliation focus KDMA",
        }
    }

    @pytest.mark.parametrize(
        ("kdma", "opt_a_value", "medical_delta", "attribute_delta", "exp_value"),
        [
            ("affiliation", 0.3, 0.7, 0.1, 0.745),
            ("merit", 0.5, 0.3, 0.2, 0.425),
            ("affiliation", 0.3, 0.7, 0.8, 0.745),  # Attribute delta shouldn't change result for affiliation/merit
            ("merit", 0.5, 0.3, 0.8, 0.425),
        ],
    )
    def test_weighted_midpoint_eqn(self, kdma, opt_a_value, medical_delta, attribute_delta, exp_value):
        """ Regression test to ensure equation doesn't get inadvertently modified """
        alignment_fn = MedicalUrgencyAlignmentWeightedADMComponent(
            TestMedicalUrgencyAlignmentWeightedADMComponent.attribute_definitions
        )

        assert (
            alignment_fn._midpoint_eqn(kdma, opt_a_value, medical_delta, attribute_delta) == pytest.approx(exp_value)
        )


class TestRandomEffectsModelAlignmentADMComponent:
    attribute_definitions = {
        "KDMA_A": {
            "name": "Merit Focus",
            "kdma": "merit",
            "description": "Test merit focus KDMA",
        },
        "KDMA_B": {
            "name": "Affiliation Focus",
            "kdma": "affiliation",
            "description": "Test affiliation focus KDMA",
        },
        "KDMA_C": {
            "name": "Personal Safety",
            "kdma": "personal_safety",
            "description": "Test personal safety KDMA",
        },
        "KDMA_D": {
            "name": "Search vs Stay",
            "kdma": "search",
            "description": "Test search vs stay KDMA",
        },
    }

    @pytest.mark.parametrize(
        ("attribute_prediction_scores", "attribute_relevance", "alignment_target", "exp_choice", "exp_raises"),
        [
            # No alignment target
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"medical": 0.6, "KDMA_A": 0.3, "KDMA_B": 0.5},
                },
                None,
                None,
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"Assumption violated: `alignment_target` was None"),
            ),
            # No medical predictions
            (
                {
                    "Choice 0": {"KDMA_A": 0.1, "KDMA_B": 0.8},
                    "Choice 1": {"KDMA_A": 0.3, "KDMA_B": 0.5},
                },
                None,
                {
                    "kdma_values":
                    [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"Medical Urgency predictions required"),
            ),
            # >2 choices
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.2, "KDMA_A": 0.3},
                    "Choice 2": {"medical": 0.6, "KDMA_A": 0.4},
                },
                None,
                {
                    "kdma_values":
                    [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(NotImplementedError, match=r"This alignment function has not yet been"),
            ),
            # <2 choices
            (
                {
                    "Choice 0": {"medical": 0.1, "KDMA_B": 0.8},
                },
                None,
                {
                   "kdma_values":
                    [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(NotImplementedError, match=r"This alignment function has not yet been"),
            ),
            # Target missing parameters
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                None,
                {
                    "kdma_values": [{"kdma": "KDMA_A", "value": 0.7}],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"This alignment function requires an intercept, medical weight, and attr weight"),
            ),
            # Target missing intercept
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                None,
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        }
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"This alignment function requires an intercept, medical weight, and attr weight"),
            ),
            # Target missing medical weight
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                None,
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        }
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"This alignment function requires an intercept, medical weight, and attr weight"),
            ),
            # Target missing attr_weight
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                None,
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                            ]
                        }
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"This alignment function requires an intercept, medical weight, and attr weight"),
            ),
            # Multiple KDMAs relevant
            (
                {
                    "Choice 0": {"medical": 0.9, "KDMA_A": 0.1},
                    "Choice 1": {"medical": 0.4, "KDMA_A": 0.9},
                },
                None,
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                        {
                            "kdma": "KDMA_B",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        }
                    ],
                },
                None,  # Raise expected so doesn't matter
                pytest.raises(RuntimeError, match=r"This alignment function can only be used when 1 attribute is relevant"),
            ),
            # Worked example with ADEPT
            (
                {
                    "Treat Patient A": {"medical": 0.947157191, "merit": 0.0},
                    "Treat Patient B": {"medical": 0.012495865, "merit": 1.0},
                    # Medical delta = 0.947157191-0.012495865 = 0.934661326
                    # Z-scaled medical delta = (0.934661326 - 0.433409) / 0.308294 = 1.62589063037
                    # Attribute score = 0.0
                    # Z-scaled attribute = (0.0 - 0.357632) / 0.27947 = -1.27967939314
                },
                None,
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.5},
                                {"name": "medical_weight", "value": 0.85},
                                {"name": "attr_weight", "value": -0.3},
                            ]
                        },
                    ],
                    # Y_ij = 0.5 + 0.85*1.62589063037-0.3*-1.27967939314 = 2.26591085376
                    # P_choose_a = e^2.26591085376/(1+e^2.26591085376) = 0.90601416437
                },
                "Treat Patient A",
                does_not_raise(),
            ),
            # Multi-KDMA, should fallback to merit
            (
                {
                    "Treat Patient A": {"medical": 0.947157191, "merit": 0.0, "affiliation": 0.5},
                    "Treat Patient B": {"medical": 0.012495865, "merit": 1.0, "affiliation": 0.25},
                    # Medical delta = 0.947157191-0.012495865 = 0.934661326
                    # Z-scaled medical delta = (0.934661326 - 0.433409) / 0.308294 = 1.62589063037
                    # Attribute score = 0.0
                    # Z-scaled attribute = (0.0 - 0.357632) / 0.27947 = -1.27967939314
                },
                {
                    "merit": 1.0,
                    "affiliation": 0.0
                },
                {
                    "kdma_values": [
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.5},
                                {"name": "medical_weight", "value": 0.85},
                                {"name": "attr_weight", "value": -0.3},
                            ]
                        },
                        {
                            "kdma": "KDMA_B",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        }
                    ],
                    # Y_ij = 0.5 + 0.85*1.62589063037-0.3*-1.27967939314 = 2.26591085376
                    # P_choose_a = e^2.26591085376/(1+e^2.26591085376) = 0.90601416437
                },
                "Treat Patient A",
                does_not_raise(),
            ),
        ],
        ids=[
            "no target", "no medical preds", ">2 choices", "<2 choices", "target missing parameters", "missing intercept",
            "missing medical weight", "missing attr weight", "multiple relevant KDMAs", "worked example", "multi-kdma"
        ],
    )
    def test_run(self, attribute_prediction_scores, attribute_relevance, alignment_target, exp_choice, exp_raises):
        alignment_fn = RandomEffectsModelAlignmentADMComponent(
            TestRandomEffectsModelAlignmentADMComponent.attribute_definitions
        )

        with exp_raises:
            # Only checking selected choice as best sample index not yet implemented
            assert alignment_fn.run(attribute_prediction_scores, alignment_target, attribute_relevance)[0] == exp_choice

    @pytest.mark.parametrize(
        ("p_choose_a", "attribute_prediction_scores", "alignment_target", "exp_choice"),
        [
            # P > 0.5
            (
                0.75,
                {  # Predictions don't matter because we are mocking compute_p_choose_a
                    "Choice 0": {"medical": 0.9, "merit": 0.1},
                    "Choice 1": {"medical": 0.4, "merit": 0.9},
                },
                {
                    "kdma_values": [  # Target doesn't matter because we are mocking compute_p_choose_a
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                "Choice 0",
            ),
            # P == 0.5
            (
                0.5,
                {  # Predictions don't matter because we are mocking compute_p_choose_a
                    "Choice 0": {"medical": 0.9, "merit": 0.1},
                    "Choice 1": {"medical": 0.4, "merit": 0.9},
                },
                {
                    "kdma_values": [  # Target doesn't matter because we are mocking compute_p_choose_a
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                "Choice 0",
            ),
            # P < 0.5
            (
                0.25,
                {  # Predictions don't matter because we are mocking compute_p_choose_a
                    "Choice 0": {"medical": 0.9, "merit": 0.1},
                    "Choice 1": {"medical": 0.4, "merit": 0.9},
                },
                {
                    "kdma_values": [  # Target doesn't matter because we are mocking compute_p_choose_a
                        {
                            "kdma": "KDMA_A",
                            "value": None,
                            "parameters": [
                                {"name": "intercept", "value": 0.75},
                                {"name": "medical_weight", "value": 0.5},
                                {"name": "attr_weight", "value": -0.25},
                            ]
                        },
                    ],
                },
                "Choice 1",
            ),
        ]
    )
    @mock.patch.object(RandomEffectsModelAlignmentADMComponent, "_compute_p_choose_a")
    def test_choice_selection(
        self, mock_compute_p_choose_a, p_choose_a, attribute_prediction_scores, alignment_target, exp_choice
    ):
        mock_compute_p_choose_a.return_value = p_choose_a

        alignment_fn = RandomEffectsModelAlignmentADMComponent(
            TestRandomEffectsModelAlignmentADMComponent.attribute_definitions
        )

        # Only checking selected choice as best sample index not yet implemented
        assert alignment_fn.run(attribute_prediction_scores, alignment_target)[0] == exp_choice

    @pytest.mark.parametrize(
        ("kdma", "intercept", "medical_weight", "attr_weight", "raw_medical_delta", "raw_attr_score", "exp_value"),
        [
            # Worked example with ADEPT
            (
                "merit", 0.5, 0.85, -0.3,
                0.934661326, 0.0,
                # Z-scaled medical delta = (0.934661326 - 0.433409) / 0.308294 = 1.62589063037
                # Z-scaled attribute = (0.0 - 0.357632) / 0.27947 = -1.27967939314
                # Y_ij = 0.5 + 0.85*1.62589063037-0.3*-1.27967939314 = 2.26591085376
                # P_choose_a = e^2.26591085376/(1+e^2.26591085376) = 0.90601416437
                0.90601416437
            ),
            (
                "affiliation", 2.1875, 2.36875, 0.015625,
                0.985162998, 0.0,
                # Z-scaled medical delta = (0.985162998 - 0.403801) / 0.297245 = 1.95583440596
                # Z-scaled attribute = (0.0 - 0.405073) / 0.298288 = -1.35799294641
                # Y_ij = 2.1875 + 2.36875*1.95583440596 + 0.015625*-1.35799294641 = 6.79916410933
                # P_choose_a = e^6.79916410933/(1+e^6.79916410933) = 0.99888653465
                0.99888653465
            ),
            (
                "personal_safety", -2.8125, 0.36875, 0.2375,
                0.312888889, 0.517996623,
                # Z-scaled medical delta = (0.312888889 - 0.456221) / 0.246484 = -0.581506755
                # Z-scaled attribute = (0.517996623 - 0.554813) / 0.303567 = -0.12127924642
                # Y_ij = -2.8125 + 0.36875*-0.581506755 + 0.2375*-0.12127924642 = -3.05573443693
                # P_choose_a = e^-3.05573443693/(1+e^-3.05573443693) = 0.04497054612
                0.04497054612
            ),
            (
                "search", 1.65, -2.34375, 0.40938,
                0.012495865, 0.312888889,
                # Z-scaled medical delta = (0.012495865 - 0.525886) / 0.357475 = -1.43615675222
                # Z-scaled attribute = (0.312888889 - 0.571051) / 0.219335 = -1.17702195728
                # Y_ij = 1.65 + -2.34375*-1.43615675222 + 0.40938*-1.17702195728 = 4.53414313914
                # P_choose_a = e^4.53414313914/(1+e^4.53414313914) = 0.98937793692
                0.98937793692
            ),
        ]
    )
    def test_compute_p_choose_a(self, kdma, intercept, medical_weight, attr_weight, raw_medical_delta, raw_attr_score, exp_value):
        alignment_fn = RandomEffectsModelAlignmentADMComponent(
            TestRandomEffectsModelAlignmentADMComponent.attribute_definitions
        )

        assert (
            alignment_fn._compute_p_choose_a(kdma, intercept, medical_weight, attr_weight, raw_medical_delta, raw_attr_score) ==
            pytest.approx(exp_value)
        )
