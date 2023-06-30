def _bin_kdma_value(value):
    # Used to map numeric KDMA values to text for injection into
    # prompt
    if value <= 0:
        return "no "
    elif value >= 3:
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
    return "\n".join(["{}".format(c['unstructured'].strip())
                      for c in casualties])


def build_prompt_options_string(options):
    options_info = ''
    options_code = 'A'
    for option in options:
        options_info += '{}: {}\n'.format(options_code, option['value'])
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

    if probe_state is not None:
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
