def calculate_votes(possible_choices,
                    responses,
                    inverse_responses=None):
    # inverse_responses here are to capture choices made with an
    # inverse prompt (i.e. prompts misaligned with the alignment
    # target)
    if len(possible_choices) != len(set(possible_choices)):
        raise RuntimeError("Possible choices for voting are not unique!")

    votes = {k: 0.0 for k in possible_choices}

    for r in responses:
        votes[r] += 1

    if inverse_responses is not None:
        for inv_r in inverse_responses:
            for choice, _ in votes.items():
                # Subtract points for inverse responses
                if inv_r == choice:
                    votes[choice] -= 1.0/len(possible_choices)
                else:
                    votes[choice] += 1.0/len(possible_choices)

    # Logic here copied over from Single KDMA ADM (variables renamed)
    # TODO: Revisit and ensure this is correct and what we want to be
    # doing; if there are standard voting schemes (and code already
    # existing for those) would likely be better to work with those
    min_score = min(votes.values()) + 1e-6
    tmp_normalized_votes = {choice: score - min_score
                            for choice, score in votes.items()}
    total = sum(tmp_normalized_votes.values())
    normalized_votes = {choice: round(score / total, 6)
                        for choice, score in tmp_normalized_votes.items()}

    return normalized_votes


def filter_votes_to_responses(votes, responses):
    filtered_votes = {choice: score for choice, score in votes.items()
                      if choice in responses}

    if len(filtered_votes) == 0:
        raise RuntimeError(
            "No votes left after filtering, was `reponses` empty?")

    return filtered_votes
