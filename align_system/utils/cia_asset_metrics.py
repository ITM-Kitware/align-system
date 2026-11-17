"""Reconstruct the Dartmouth tied-asset impact metric from saved Red actions.

Matches ResilienceMetric with no blue_action and the original scorer mappings.
These are event penalties, not state-based measures of remaining asset health.
"""

ASSET_IMPACT_WEIGHTS = {
    'Auth': (10.0, 10.0, 10.0),
    'Database': (10.0, 0.0, 10.0),
    'Front': (0.0, 10.0, 10.0),
}


def asset_impact_scores(red_action):
    """Return positive C/I/A penalties for a successful Impact on a tied asset.

Use the saved Red outcome even if Blue also restored the host that step.
ExploitRemoteService, PrivilegeEscalate and failed Impact actions score zero.
"""
    if red_action['Action'] != 'Impact' or red_action['Status'] != 'TRUE':
        return [0.0, 0.0, 0.0]
    host = red_action['Host']
    # Match ImpactReporter's host classification and precedence exactly.
    for asset, weights in ASSET_IMPACT_WEIGHTS.items():
        if asset in host:
            return list(weights)
    return [0.0, 0.0, 0.0]
