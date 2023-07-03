from typing import List


def force_choice(text: str, choices: List[str], similarity_measure_func):
    top_score = -float('inf')
    top_choice = None
    top_choice_idx = None
    for i, choice in enumerate(choices):
        score = similarity_measure_func(text, choice)

        # Assumes higher score is better match
        # TODO: Add option to prioritize lower score instead of higher
        if score > top_score:
            top_score = score
            top_choice = choice
            top_choice_idx = i

    return top_choice_idx, top_choice
