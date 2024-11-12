from collections.abc import MutableMapping


def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def standardize_injury_loc(injury):
    if injury is None:
        return None
    return injury.replace('right ', '').replace('left ', '').replace('center ', '')


military_disposition_mapping = {'Military Adversary': 0,
                                'Non-Military Adversary': 1,
                                'Civilian': 2,
                                'Allied': 3,
                                'Allied US': 4,
                                None: -1}

rapport_mapping = {'dislike': 0,
                   'neutral': 1,
                   'close': 2,
                   None: -1}

intent_mapping = {'intend major harm': 0,
                  'intend minor harm': 1,
                  'no intent': 2,
                  'intend minor help': 3,
                  'intend major help': 4,
                  None: -1}

doc_mapping = {'indirect': 0,
               'somewhat indirect': 1,
               'none': 2,
               'somewhat direct': 3,
               'direct': 4,
               None: -1}

injury_type_mapping = {'Chest Collapse': 0,
                       'Shrapnel': 1,
                       'Asthmatic': 2,
                       'Puncture': 3,
                       'Traumatic Brain Injury': 4,
                       'Broken Bone': 5,
                       'Abrasion': 6,
                       'Burn': 7,
                       'Open Abdominal Wound': 8,
                       'Amputation': 9,
                       'Laceration': 10,
                       'Internal': 11,
                       'Ear Bleed': 12,
                       None: -1}

injury_loc_mapping = {'thigh': 0,
                      'face': 1,
                      'forearm': 2,
                      'chest': 3,
                      'calf': 4,
                      'shoulder': 5,
                      'neck': 6,
                      'head': 7,
                      'bicep': 8,
                      'internal': 9,
                      'wrist': 10,
                      'leg': 11,
                      'side': 12,
                      'stomach': 13,
                       None: -1}

injury_sev_mapping = {'minor': 0,
                      'moderate': 1,
                      'substantial': 2,
                      'major': 3,
                      'extreme': 4,
                       None: -1}

treatment_type_mapping = {'Blood': 0,
                          'Burn Dressing': 1,
                          'Hemostatic gauze': 2,
                          'Nasopharyngeal airway': 3,
                          'Pressure bandage': 4,
                          'Tourniquet': 5,
                          None: -1}

action_type_mapping = {'APPLY_TREATMENT': 0,
                       'CHECK_ALL_VITALS': 1,
                       'CHECK_BLOOD_OXYGEN': 2,
                       'CHECK_PULSE': 3,
                       'CHECK_RESPIRATION': 4,
                       'END_SCENE': 5,
                       'MESSAGE': 6,
                       'MOVE_TO': 7,
                       'MOVE_TO_EVAC': 8,
                       'SEARCH': 9,
                       None: -1}
