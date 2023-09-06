from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import re


def _filtered_to_cuda(model: nn.Module) -> None:
    """Does model.cuda(), but doesn't reassign tensors that are
    already on gpu"""

    def conditional_cuda(x):
        if x.device.type != "cuda":
            return x.to("cuda:0")
        else:
            return x

    return model._apply(conditional_cuda)


PATIENT_ID_REGEXP = re.compile(r'patient_[0-9a-f]{8}\b-[0-9a-f]{4}\b-[0-9a-f]{4}\b-[0-9a-f]{4}\b-[0-9a-f]{12}', re.I)  # noqa


def select_first_mentioned_patient(generated_text):
    mentioned_patients = re.findall(PATIENT_ID_REGEXP, generated_text)

    if len(mentioned_patients) == 0:
        return None
    else:
        return mentioned_patients[0]


# May need to pass additional information (e.g. patient info)
def prepare_prompt(scenario, probe, alignment_target=None):
    scenario_description = scenario.state.unstructured
    question = probe.prompt

    if alignment_target is not None:
        question = "{}; {}".format(
            _build_alignment_target_string(alignment_target.to_dict()),
            question).capitalize()

    return f"""
Here is a scenario related to diaster relief. We need to find the next step with right moral reasoning and give reason why : {scenario_description}

{question}"""


# May need to pass additional information (e.g. patient info)
def prepare_prompt_instruct_gpt_j(scenario, probe, alignment_target=None):
    scenario_description = scenario.state.unstructured
    question = probe.prompt

    if alignment_target is not None:
        question = "{}; {}".format(
            _build_alignment_target_string(alignment_target.to_dict()),
            question).capitalize()

    return f"""
Here is a scenario related to diaster relief : {scenario_description}

{question}"""


def _bin_kdma_value(value):
    # Used to map numeric KDMA values to text for injection into
    # prompt
    if value <= 0:
        return "no "
    elif value >= 3:
        return "heavy "
    else:
        return ""


def _build_alignment_target_string(alignment_target):
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


class LLMBaseline:
    def __init__(self, device="cuda", model_use="gpt-j", distributed=False):
        self.device = device
        self.model_use = model_use
        self.distributed = distributed

        self.model = None
        self.tokenizer = None
        self.model_loaded = False

    def load_model(self):
        if self.model_use == 'gpt-j':
            self.model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-j-6B")
        if self.model_use == 'instruct-gpt-j':
            self.model = AutoModelForCausalLM.from_pretrained(
                "nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpcloud/instruct-gpt-j-fp16")
        if self.model_use == 'falcon':
            # self.model_name = 'tiiuae/falcon-7b-instruct'
            self.model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b-instruct",
                # torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tiiuae/falcon-7b-instruct"
            )

        if self.distributed:
            self.model = _filtered_to_cuda(self.model.half())
        else:
            self.model.to(self.device)

        self.model_loaded = True

    def run_inference(self, prompt):
        # Check if the tokenizer has a pad token; if not, set it to the eos_token or add a new token
        if not self.tokenizer.pad_token:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})

        encoding = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()

        len_context = input_ids.shape[-1] + 256

        gen_tokens = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.001,
            max_length=len_context,
            pad_token_id=self.tokenizer.pad_token_id
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

        return gen_text


