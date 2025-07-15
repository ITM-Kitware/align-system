import pytest
from contextlib import nullcontext as does_not_raise

from align_system.algorithms.alignment_adm_component import (
    MedicalUrgencyAlignmentADMComponent,
    MedicalUrgencyAlignmentWeightedADMComponent)

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
            # Fully tied patients would be random choice,
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
                "Choice 0",  # KDMA_A target is exactly midpoint so its votes don't count
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
                "Choice 1",  # KDMA_A target is exactly midpoint so its votes don't count
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
