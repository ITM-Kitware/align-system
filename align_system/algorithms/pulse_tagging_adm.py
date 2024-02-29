from align_system.algorithms.abstracts import AlignedDecisionMaker
from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel


def convert_to_string(data):
    # Mapping for full names
    full_names = {
        "HR_bpm": "Heart Rate",
        "RR_bpm": "Respiration Rate",
        "SBP_mmHg": "Systolic Blood Pressure",
        "CoreTemp_C": "Core Body Temperature",
        "PPI": "Pulse Pressure Index",
        "BO2PP_mmHg": "Blood Oxygen Partial Pressure",
        "GCS": "Glasgow Coma Scale",
        "SkinColor": "Skin Color",
        "SkinTouch": "Skin Touch",
        "CRT_s": "Capillary Refill Time",
        "RadialPulsePalpability": "Radial Pulse Palpability"
    }

    # Convert and round the values
    result = []
    for key, value in data.items():
        name = full_names.get(key, key)
        if isinstance(value, float):
            value = round(value, 3)  # Adjust the rounding as needed
        result.append(f"{name}: {value}")

    return "\n".join(result)

class PulseTaggingADM(ChatLanguageModel, AlignedDecisionMaker):
    
    
    def identify_tag_color(self, message):
        chat_response = self.generate_responses(
            dialogs=[[
                {
                    'role': 'system',
                    'content': 'Format the user\'s message as a JSON object identifying the tag color. GREEN, YELLOW, RED, BLACK'
                },
                # black tag example
                {
                    'role': 'user',
                    'content': 'Because the casualty is not breathing, they should be tagged as **EXPECTANT** (Black Triage Tag Color), indicating that they are unlikely to survive given the severity of injuries, level of available care, or both. Palliative care and pain relief should be provided.'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "BLACK" }'
                },
                # green tag example
                {
                    'role': 'user',
                    'content': 'They should be tagged GREEN'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "GREEN" }'
                },
                # yellow tag example
                {
                    'role': 'user',
                    'content': 'Due to the severity of the casualty\'s injuries, they should be tagged YELLOW'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "YELLOW" }'
                },
                # red tag example
                {
                    'role': 'user',
                    'content': 'The casualty is not breathing, but has a pulse. They should be tagged RED'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "RED" }'
                },
                # actual example
                {
                    'role': 'user',
                    'content': message
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color":'
                }
            ]]
        )[0]
        for i, color in enumerate(['GREEN', 'YELLOW', 'RED', 'BLACK']):
            if color in chat_response.upper():
                return i
            
        for i, color in enumerate(['MINOR', 'DELAYED', 'IMMEDIATE', 'EXPECTANT']):
            if color in chat_response.upper():
                return i
        
        return -1
    
    def __call__(self, sample, target_kdma_values, **kwargs):
        # The start_guide is a text formatting of the START guide flowchart
        # TODO move these strings to template filess
        start_guide = '''**START Adult Triage Guide for Tagging Casualties**

1. **Initial Assessment: Ability to Walk**
   - If the casualty is able to walk, they are tagged as **MINOR** (Green Triage Tag Color) and directed to secondary triage.

2. **Spontaneous Breathing Check**
   - If the casualty is not breathing even after the airway is positioned, they are tagged as **EXPECTANT** (Black Triage Tag Color), indicating they are unlikely to survive given the severity of injuries, level of available care, or both. Palliative care and pain relief should be provided.
   - If the casualty resumes spontaneous breathing after positioning the airway, proceed to the next check.

3. **Respiratory Rate Check**
   - If the respiratory rate is greater than or equal to 30 breaths per minute, tag the casualty as **IMMEDIATE** (Red Triage Tag Color), indicating the need for immediate intervention and transport.

4. **Perfusion Check**
   - Assess the radial pulse or capillary refill time. If the radial pulse is absent or capillary refill time is greater than 2 seconds, tag the casualty as **IMMEDIATE**.
   - If the radial pulse is present or capillary refill time is less than 2 seconds, proceed to the next check.

5. **Mental Status Check**
   - If the casualty does not obey commands, tag them as **IMMEDIATE**.
   - If the casualty obeys commands, tag them as **DELAYED** (Yellow Triage Tag Color), indicating that their transport can be delayed and they have serious but potentially not life-threatening injuries, and are not expected to deteriorate significantly over several hours.

**Additional Information on Triage Categories:**
- **EXPECTANT:** Victim unlikely to survive. Provide palliative care.
- **IMMEDIATE:** Victim needs immediate help. Includes medical attention within minutes for survival.
- **DELAYED:** Serious injuries, but not life-threatening and not expected to deteriorate soon.
- **MINOR:** Minor injuries. May be able to assist in their own care.
'''
        sample_1 = 'Patient is able to walk to the designated safety area.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is able to walk, so they can be tagged as MINOR (Green).

2. Spontaneous Breathing Check

The patient is able to walk, so they can be tagged as MINOR (Green).

3. Respiratory Rate Check

The patient is able to walk, so they can be tagged as MINOR (Green).

4. Perfusion Check

The patient is able to walk, so they can be tagged as MINOR (Green).

5. Mental Status Check

The patient is able to walk, so they can be tagged as MINOR (Green).

Conclusion: Tag the casualty as MINOR (Green).'''

        sample_2 = 'The patient is conscious. Patient has a moderate hemorrhage. You have used a tourniquet to stop the bleeding. Patient is breathing normally. Patient pulse is within the normal range. The patient is able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

It is unclear if the patient is able to walk, so I will move on to the next check.

2. Spontaneous Breathing Check

The patient is breathing normally, so they so they should not be tagged as EXPECTANT (Black).

3. Respiratory Rate Check

The patient is breathing normally, which is less than 30 breaths per minute, so I don't need to tag them as IMMEDIATE (Red) just yet.

4. Perfusion Check

The patient's pulse is within the normal range, so they will either be tagged RED or YELLOW depending on the Mental Status Check.

5. Mental Status Check

The casualty follows basic commands, so they should be tagged as DELAYED (Yellow)

Conclusion: Tag the casualty as DELAYED (Yellow)'''

        sample_3 = 'The patient is conscious. Patient is breathing normally. Patient pulse is abnormally high. Capillary refill time is more than 3 seconds. The patient is not able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is not able to walk so they cannot be tagged as MINOR (Green).

2. Spontaneous Breathing Check

The patient is breathing normally so they cannot be tagged as EXPECTANT (Black).

3. Respiratory Rate Check

The patient's respiratory rate is not faster than 30 breaths per minute, so it is possible they should be tagged YELLOW.

4. Perfusion Check

The patient's pulse is abnormally high and capillary refill time is more than 3 seconds, so they do not pass this check which means they should be tagged as IMMEDIATE (Red).

5. Mental Status Check

The patient is not able to follow basic commands, so they do not pass this check.

Conclusion: Tag the casualty as IMMEDIATE (Red)'''

        sample_4 = 'The patient is unconscious. Patient is not breathing. Repositioning the airway did not help. Patient has a moderate hemorrhage. You have used a tourniquet to stop the bleeding. No radial pulse can be detected on the patient. Capillary refill time is more than 3 seconds. The patient is not able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is unconscious and not able to walk, so they should not be tagged as MINOR (Green).

2. Spontaneous Breathing Check

The patient is not breathing, so they do not pass this check.

3. Respiratory Rate Check

The patient is not breathing, so they do not pass this check.

4. Perfusion Check

The patient has no radial pulse and capillary refill time is more than 3 seconds, so they do not pass this check.

5. Mental Status Check

The patient is unconscious and not able to follow basic commands, so they do not pass this check.

Conclusion: Tag the casualty as EXPECTANT (Black)'''

        dialogs = [[
            {
                'role': 'system',
                'content': start_guide
            },
            {
                'role': 'user',
                'content': sample_3[0]
            },
            {
                'role': 'assistant',
                'content': sample_3[1]
            },
            {
                'role': 'user',
                'content': sample_1[0]
            },
            {
                'role': 'assistant',
                'content': sample_1[1]
            },
            {
                'role': 'user',
                'content': sample_2[0]
            },
            {
                'role': 'assistant',
                'content': sample_2[1]
            },
            {
                'role': 'user',
                'content': sample_3[0]
            },
            {
                'role': 'assistant',
                'content': sample_3[1]
            },
            {
                'role': 'user',
                'content': sample['scenario']
            },
            {
                'role': 'assistant',
                'content': 'Let\'s apply the START guide'
            }
        ]]
        
        
        
        n_samples = kwargs.get('n_samples', 1)  # Set the number of samples, default to 5 if not provided

        chat_responses = []  # List to store the generated responses

        
        chat_responses = self.generate_responses(dialogs * n_samples)
            # chat_responses.append(response)

        color_counts = {}  # Dictionary to store the count of each color_idx

        for response in chat_responses:
            color_idx = self.identify_tag_color(response)
            color_counts[color_idx] = color_counts.get(color_idx, 0) + 1

        most_popular_color_idx = max(color_counts, key=color_counts.get)  # Get the most popular color_idx

        most_popular_reasoning = None

        for response in chat_responses:
            color_idx = self.identify_tag_color(response)
            if color_idx == most_popular_color_idx:
                most_popular_reasoning = response
                break

        return {
            'choice': most_popular_color_idx,
            'info': {
                'generated_reasoning': most_popular_reasoning,
                'color_counts': color_counts,
                'all_reasonings': chat_responses,
            }
        }