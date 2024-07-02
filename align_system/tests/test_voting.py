import pytest

from align_system.utils.voting import calculate_votes
# from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM


def test_calculate_votes_1():
    choices = ['a', 'b', 'c']
    responses = ['a']

    votes = calculate_votes(choices, responses)

    assert votes == {'a': pytest.approx(1.0, 1e-5),
                     # need to use abs for expected value of 0
                     'b': pytest.approx(0.0, abs=1e-5),
                     'c': pytest.approx(0.0, abs=1e-5)}


def test_calculate_votes_2():
    choices = ['a', 'b', 'c']
    responses = ['a', 'a', 'b']

    votes = calculate_votes(choices, responses)

    assert votes == {'a': pytest.approx(2/3, 1e-5),
                     'b': pytest.approx(1/3, 1e-5),
                     'c': pytest.approx(0.0, abs=1e-5)}


def test_calculate_votes_3():
    choices = ['a', 'b', 'c']
    inverse_responses = ['b', 'b', 'c']

    votes = calculate_votes(choices, [],
                            inverse_responses=inverse_responses)

    assert votes == {'a': pytest.approx((4/3) / 2, 1e-5),
                     'b': pytest.approx(0 / 2, abs=1e-5),
                     'c': pytest.approx((2/3) / 2, 1e-5)}


def test_calculate_votes_4():
    choices = ['a', 'b', 'c']
    responses = ['a', 'a', 'b', 'c', 'c']
    inverse_responses = ['b', 'b', 'c']

    votes = calculate_votes(choices, responses,
                            inverse_responses=inverse_responses)

    assert votes == {'a': pytest.approx((7/3) / 4, 1e-5),
                     'b': pytest.approx(0 / 4, abs=1e-5),
                     'c': pytest.approx((5/3) / 4, 1e-5)}


# def test_old_calculate_votes_1():
#     choices = ['a', 'b', 'c']

#     responses = [{'answer_idx': 0, 'aligned': True}]

#     votes = Llama2SingleKDMAADM.calculate_votes(responses, choices)

#     assert votes == [pytest.approx(1.0, 1e-5),
#                      pytest.approx(0.0, abs=1e-5),
#                      pytest.approx(0.0, abs=1e-5)]


# def test_old_calculate_votes_2():
#     choices = ['a', 'b', 'c']

#     responses = [{'answer_idx': 0, 'aligned': True},
#                  {'answer_idx': 0, 'aligned': True},
#                  {'answer_idx': 1, 'aligned': True}]

#     votes = Llama2SingleKDMAADM.calculate_votes(responses, choices)

#     assert votes == [pytest.approx(2/3, 1e-5),
#                      pytest.approx(1/3, abs=1e-5),
#                      pytest.approx(0.0, abs=1e-5)]


# def test_old_calculate_votes_3():
#     choices = ['a', 'b', 'c']

#     responses = [{'answer_idx': 1, 'aligned': False},
#                  {'answer_idx': 1, 'aligned': False},
#                  {'answer_idx': 2, 'aligned': False}]

#     votes = Llama2SingleKDMAADM.calculate_votes(responses, choices)

#     assert votes == [pytest.approx((4/3)/2, 1e-5),
#                      pytest.approx(0/2, abs=1e-5),
#                      pytest.approx((2/3)/2, abs=1e-5)]


# def test_old_calculate_votes_4():
#     choices = ['a', 'b', 'c']

#     responses = [{'answer_idx': 0, 'aligned': True},
#                  {'answer_idx': 0, 'aligned': True},
#                  {'answer_idx': 1, 'aligned': True},
#                  {'answer_idx': 2, 'aligned': True},
#                  {'answer_idx': 2, 'aligned': True},

#                  {'answer_idx': 1, 'aligned': False},
#                  {'answer_idx': 1, 'aligned': False},
#                  {'answer_idx': 2, 'aligned': False}]

#     votes = Llama2SingleKDMAADM.calculate_votes(responses, choices)

#     assert votes == [pytest.approx((7/3) / 4, 1e-5),
#                      pytest.approx(0 / 4, abs=1e-5),
#                      pytest.approx((5/3) / 4, 1e-5)]
