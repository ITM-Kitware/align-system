from swagger_client.models import InjuryStatusEnum


def _bin_kdma_value(value):
    # Used to map numeric KDMA values to text for injection into
    # prompt
    if value <= 0:
        return "no "
    elif value > 5:
        return "heavy "
    else:
        return ""


def build_alignment_target_string(alignment_target):
    *kdma_values, last_kdma_value = alignment_target['kdma_values']

    alignment_target_string_pieces = ["with"]
    for a in kdma_values:
        alignment_target_string_pieces.append(" {}emphasis on {}, ".format(
            _bin_kdma_value(a['value']), a['kdma']))

    if len(kdma_values) > 0:
        alignment_target_string_pieces.append('and')

    alignment_target_string_pieces.append(" {}emphasis on {}".format(
        _bin_kdma_value(last_kdma_value['value']),
        last_kdma_value['kdma']))

    return ''.join(alignment_target_string_pieces)


def build_casualties_string(casualties):
    return "\n".join(["{}: {}".format(c['id'], c['unstructured'].strip())
                      for c in casualties])


def build_casualties_string_with_vitals(casualties):
    casualty_strings = []

    for casualty in casualties:
        casualty_vitals = casualty.get('vitals', {})
        if len(casualty_vitals) > 0:
            casualty_vitals_str =\
                "\n".join(["{}: {}".format(k, v)
                           for k, v in casualty_vitals.items()
                           if v is not None])
        else:
            casualty_vitals_str = ""

        casualty_strings.append(f"""
{casualty.get('unstructured', '').strip()}
{casualty_vitals_str}
""")

    return "{}\n".format("\n".join(casualty_strings))


def build_prompt_options_string(options):
    options_info = ''
    options_code = 'A'
    for option in options:
        options_info += '{}: {}\n'.format(options_code, option['value'])
        options_code = chr(ord(options_code) + 1)

    return options_info


def build_prompt_available_actions_string(available_actions):
    options_info = ''
    options_code = 'A'
    for option in available_actions:
        options_info += '{}: {}\n'.format(options_code, option)
        options_code = chr(ord(options_code) + 1)

    return options_info


def prepare_prompt(scenario_info,
                   scenario_mission,
                   probe_state,
                   probe_prompt,
                   casualties,
                   options=None,
                   alignment_target=None):

    if alignment_target is not None:
        alignment_target_string = " {}".format(
            build_alignment_target_string(alignment_target))
    else:
        alignment_target_string = ""

    if options is not None:
        ammended_prompt = "{} Choose from the following:\n{}".format(
            probe_prompt, build_prompt_options_string(options))
    else:
        ammended_prompt = probe_prompt

    if scenario_info is not None:
        scenario_info_str = "{}".format(scenario_info)
    else:
        scenario_info_str = ""

    if scenario_mission is not None:
        scenario_mission_str = "{}".format(scenario_mission)
    else:
        scenario_mission_str = ""

    if probe_state is not None and (probe_state.strip()
                                    != scenario_info.strip()):
        probe_state_str = "{}\n".format(probe_state)
    else:
        probe_state_str = ""

    casualties_str = "{}\n".format(build_casualties_string(casualties))

    return f"""
Instruction: Use medical triage knowledge to answer the following question{alignment_target_string}.
{scenario_info_str}{casualties_str}
Mission for the task is : {scenario_mission_str}
{probe_state_str}{ammended_prompt}
Response: """  # noqa


def prepare_action_based_prompt(scenario_info,
                                scenario_mission,
                                current_state,
                                casualties,
                                available_actions=None,
                                alignment_target=None):

    if alignment_target is not None:
        alignment_target_string = " {}".format(
            build_alignment_target_string(alignment_target))
    else:
        alignment_target_string = ""

    if available_actions is None:
        probe_prompt = "What action should you take first?"
    else:
        probe_prompt = "Choose one of the following actions:\n{}".format(
            build_prompt_available_actions_string(available_actions))

    if scenario_info is not None:
        scenario_info_str = "{}".format(scenario_info)
    else:
        scenario_info_str = ""

    if scenario_mission is not None:
        scenario_mission_str = "{}".format(scenario_mission)
    else:
        scenario_mission_str = ""

    if current_state is not None and (current_state.strip()
                                      != scenario_info.strip()):
        current_state_str = "{}\n".format(current_state)
    else:
        current_state_str = ""

    casualties_str = "{}\n".format(build_casualties_string(casualties))

    return f"""
Instruction: Use medical triage knowledge to answer the following question{alignment_target_string}.
{scenario_info_str}
Mission for the task is : {scenario_mission_str}
Casualties:
{casualties_str}
{current_state_str}{probe_prompt}

Response: """  # noqa


def prepare_treatment_selection_prompt(casualty_description,
                                       casualty_vitals,
                                       casualty_injuries,
                                       available_supplies):
    if len(casualty_vitals) > 0:
        casualty_vitals_str = "{}\n".format(
            "\n".join(["{}: {}".format(k, v)
                       for k, v in casualty_vitals.items()
                       if v is not None]))
    else:
        casualty_vitals_str = ""

    supplies_str = "{}\n".format(
        '\n'.join(["{}: {} ({})".format(chr(i + 65), s['type'], s['quantity'])
                   for i, s in enumerate(available_supplies)]))

    untreated_injuries = [i for i in casualty_injuries if i['status'] != InjuryStatusEnum.TREATED]
    if len(untreated_injuries) > 0:
        injuries_str = "Injuries:\n{}".format(
            '\n'.join(["A {} {} of the {}".format(i['severity'], i['name'], i['location'])
                       for i in untreated_injuries]))
    else:
        injuries_str = ""

    return f"""
In a medical triage situation, you're treating the following casualty:
{injuries_str}

{casualty_vitals_str}

The medical supplies you have available to you are:
{supplies_str}

Select just one of the medical supplies to use to treat the casualty, and specify where on the body the treatment should be applied.

Response: """  # noqa


def prepare_tagging_selection_prompt(casualties, tags):
    casualties_string = build_casualties_string_with_vitals(casualties)

    tags_str = "{}\n".format('\n'.join(tags))

    return f"""
In a medical triage situation, you need to assign a triage tag one of the following casualty:
{casualties_string}

The possible medical triage tags you can apply are:
{tags_str}

Which specific casualty should you tag, and what triage tag should be applied to the casualty?

Response: """  # noqa


def prepare_character_selection_prompt(action):
    action_str = action.unstructured.lower().replace('_', ' ')

    return f"""
In a medical triage situation, you're performing the following action: {action_str}.

Which casualty should the action apply to?

Response: """  # noqa
