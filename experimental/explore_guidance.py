import guidance
import sys
import torch

from transformers import AutoTokenizer


'''

```bash
#!/bin/bash
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=priority
#SBATCH --account=itm
#SBATCH --mem 30000
#SBATCH --output=outputs/explore_guidance.md

python experimental/explore_guidance.py
```

'''

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

guidance.llm = guidance.llms.Transformers(
    model=model, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id
)

def execute_program(question, options):
    answer_select_string = "{{#select 'answer'}}"
    for option in options:
        answer_select_string += option + "{{or}}"
    answer_select_string = answer_select_string[:-len("{{or}}")]
    answer_select_string += "{{/select}}"

    prompt = """Answer the following question by writing one line explaining your thought process and then identifying the correct answer.
    Question: {{question}}
    Options: {{options}}
    Explanation: {{gen 'explanation' stop="\\n"}}
    Answer Index: """
    prompt += answer_select_string

    print(prompt)

    program = guidance(prompt)

    return program(
        question=question,
        options=str(options)
    )

def show_qa(question, options):
    executed_program = execute_program(question, options)

    # print explanation and answer
    print("## Question:")
    print(executed_program['question'])

    print("## Options:")
    print(executed_program['options'])
    print("## Response:")
    print(f"**E:** {executed_program['explanation']}")
    print(f"**A:** {executed_program['answer']}")

if __name__ == "__main__":
    question1 = (
        "After an IED explosion, you are called over to a wounded civilian who is not ambulatory,  is lying on his side, and has minor burns to the arms as well as bruising to the head and neck.  Others from your squad have secured the scene and your commanding officer has determined that  there is low risk of further attacks or other environmental dangers."
        "A 25 YO male civilian hit by an IED.  Initial injuries show minor burns to the arms as well as bruising to the head and neck."
        "What is the first thing you would do upon arriving to the scene?"
    )

    options1 = [
        "Check for consciousness",
        "Look, listen, and feel for breathing",
        "Tag civilian RED for Immediate",
        "Check for mental status"
    ]

    show_qa(question1, options1)

    question3 = (
        "After an IED explosion, you are called over to a wounded civilian who is not ambulatory,  is lying on his side, and has minor burns to the arms as well as bruising to the head and neck.  Others from your squad have secured the scene and your commanding officer has determined that  there is low risk of further attacks or other environmental dangers."
        "A 25 YO male civilian hit by an IED.  Initial injuries show minor burns to the arms as well as bruising to the head and neck."
        "Mission for the task is : Ensure that the wounded civilian(s) are treated for their injuries."
        "The civilian is conscious and lying on his back with arms at his sides.  You look, listen, and feel for breathing and find that he is breathing."
        "How fast of a breathing rate (breaths per minute) would warrant a RED (IMMEDIATE) tag? More than ___ breaths per minute."
    )

    options3 = [
        "20",
        "30",
        "40",
        "50",
    ]

    show_qa(question3, options3)