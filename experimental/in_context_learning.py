import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from algorithms import llm_baseline

'''

```bash
#!/bin/bash
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=priority
#SBATCH --account=itm
#SBATCH --mem 30000
#SBATCH --output=outputs/in_context_learning.md

python experimental/in_context_learning.py
```

'''

# for experimentation with encoding prompts
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


scenario_file = '/data/david.joy/ITM/data/bbn_adept_demo_data/scenario.json'
probe_files = [
    '/data/david.joy/ITM/data/bbn_adept_demo_data/probe1.json',
    '/data/david.joy/ITM/data/bbn_adept_demo_data/probe2.json',
    '/data/david.joy/ITM/data/bbn_adept_demo_data/probe3.json',
    '/data/david.joy/ITM/data/bbn_adept_demo_data/probe4.json'
]

with open(scenario_file) as f:
    scenario = json.load(f)

probes = []
for probe_file in probe_files:
    probes.append(json.load(open(probe_file)))


for probe in probes:
    for option in probe['options']:
        if 'Knowledge' in option['kdma_association']:
            option['kdma_association']['knowledge'] = option['kdma_association']['Knowledge']
            del option['kdma_association']['Knowledge']


answers = sum(
    [
        [
            option['value']
            for option in probe['options']
            if option['kdma_association']['knowledge'] == 3
        ]
        for probe in probes
    ],
    start=[]
)



def build_prompt(scenario, probe, answer=None):
    state = scenario['state']['unstructured'].strip()
    mission = scenario['state']['mission']['unstructured'].strip()
    supplies = scenario['state']['supplies']
    if supplies is None:
        supplies = '[]'
    else:
        supplies = json.dumps(supplies)
    
    casualties = [casualty['unstructured'].strip() for casualty in scenario['state']['casualties']]
    casualties = json.dumps(casualties)

    question = probe['prompt']

    options = [str(option['value']).strip() for option in probe['options']]


    prompt = (
        'You are a battlefield medic. You have been deployed to the following scenario:\n'
        f'{state}\n'
        f'{mission}\n'
        f'Supplies: {supplies}\n'
        f'Casualties: {casualties}\n'
        f'Q: {question}\n'
        f'Options: {options}\n'
        'Identify the correct option from the list of options.\n'
        'A:' + (f' {answer}' if answer is not None else '')
    )

    if answer is not None:
        prompt += '<|endoftext|>'

    return prompt, options


def build_example(probe, prefix, question, answer):
    options = [str(option['value']).strip() for option in probe['options']]


    prompt = (
        f'{prefix}\n'
        f'Q: {question}\n'
        f'Options: {options}\n'
        'Identify the correct option from the list of options.\n'
        'A:' + (f' {answer}' if answer is not None else '')
    )

    if answer is not None:
        prompt += '<|endoftext|>'


    return prompt, options


paraphrased_prefix = '''You assume the role of a medic in a conflict zone. You have been summoned to a situation involving the following:
A non-walking civilian victim has fallen prey to an IED blast, showing symptoms of light burns on the arms along with contusions on his neck and cranium. The place of the incident has been contained by your fellow squad members and under the review of your superior officer, it has been evaluated as a low-threat setting regarding future ambushes or external hazards.
Your responsibility lies in providing medical aid to the civilian victim(s).
Resources: []
Victim impact: ["A civilian male around 25 years of age victimized by an IED. The initial diagnosis displays light burns on the arms and contusions on the head and neck."]'''

paraphrased_questions = [
    'What would be your immediate steps upon reaching the location?',
    'What breathing frequency (inhalations per minute) would justify a RED (URGENT) label? Over ___ inhalations each minute.',
    'What does a respiration rate of 20 breaths per minute indicate regarding the condition of the civilian?',
    'What method would you employ to detect any bleeding under these circumstances?',
]


if __name__ == '__main__':
    llm = llm_baseline.LLMBaseline()
    llm.load_model()

    offset = 0

    for i, (probe, answer) in list(enumerate(zip(probes, answers)))[offset:]:
        prompt = build_prompt(scenario, probes[i-1], answers[i-1])[0]
        # prompt = build_example(probe, paraphrased_prefix, paraphrased_questions[i-offset], answer)[0]
        # embeddings = model.encode(prompt)
        # print(embeddings.shape)
        prompt += '\n\n'
        prompt += build_prompt(scenario, probe)[0]
        response = llm.run_inference(prompt)
        print('-'*10)
        print(response)
