from swagger_client.models import (
    State,
    Action,
    Character,
    Supplies,
    Injury,
    Environment,
    DecisionEnvironment,
    SimEnvironment,
    MetaInfo,
)


def hydrate_scenario_state(record):
    """ Hydrate scenario state from record """
    state = State(**record['full_state'])
    state.meta_info = MetaInfo(**state.meta_info)
    # For some reason this initialization from a dictionary
    # doesn't recursively init; need to manually do it
    state.characters = [Character(**c) for c in state.characters]
    for c in state.characters:
        c.injuries = [Injury(**i) for i in c.injuries]
    state.supplies = [Supplies(**s) for s in state.supplies]
    state.environment = Environment(**state.environment)
    state.environment.decision_environment = DecisionEnvironment(
        **state.environment.decision_environment)
    state.environment.sim_environment = SimEnvironment(
        **state.environment.sim_environment)

    actions = [Action(**a) for a in record['choices']]
    # TODO: Fix this on the input-output generation side, need
    # to make sure original choices aren't being modified by
    # ADM; for now manually clearing the justification string
    for a in actions:
        a.justification = None

    return state, actions
