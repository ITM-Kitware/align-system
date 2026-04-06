import json
from typing import List

import outlines
from outlines import Template as OutlinesTemplate
from pydantic import BaseModel


class KdmaScore(BaseModel):
    description: str
    kdma_value: float


class KdmaScoreList(BaseModel):
    entries: List[KdmaScore]


class GeneratedSystemPromptTemplate:
    def __init__(
        self,
        kdma_name,
        dataset,
        template_prompt,
        kdma_scale_factor=1.0,
        prompt_header=None,
        prompt_footer=None,
        preprocess_fn=None,
        chunk_size=512,
    ):
        from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state

        self.state_hydration_fn = p2triage_hydrate_scenario_state

        self.kdma_name = kdma_name

        self.template_prompt = OutlinesTemplate.from_string(template_prompt)

        self.kdma_scale_factor = kdma_scale_factor

        self.prompt_header = prompt_header
        self.prompt_footer = prompt_footer

        self.preprocess_fn = preprocess_fn

        self.dataset = self._load_dataset(dataset)

        self.chunk_size = chunk_size

        self.system_prompt = None

    def __call__(self, model):
        if self.system_prompt is not None:
            return self.system_prompt

        def format_entries(entries, scale=1.0):
            return "\n".join(
                f"{x['description']} would score {x['kdma_value'] * scale}"
                for x in entries
            )

        generator = outlines.Generator(model, KdmaScoreList)

        accumulated = []
        for i in range(0, len(self.dataset), self.chunk_size):
            chunk_data = accumulated + self.dataset[i : i + self.chunk_size]
            data = format_entries(chunk_data)
            prompt = self.template_prompt(dataset=data)
            result = generator(prompt, max_new_tokens=8192)
            accumulated = KdmaScoreList.model_validate_json(result).model_dump()[
                "entries"
            ]

        parts = [self.prompt_header] if self.prompt_header else []
        parts += [format_entries(accumulated, self.kdma_scale_factor)]
        parts += [self.prompt_footer] if self.prompt_footer else []
        self.system_prompt = "\n".join(parts)
        return self.system_prompt

    def _load_dataset(self, dataset):
        key = "medical_condition" if self.kdma_name == "medical" else "attribute_rating"

        with open(dataset, mode="rb") as f:
            dataset = json.load(f)

        results = []
        seen = set()
        for icl_sample in dataset:
            state, actions = self.state_hydration_fn(icl_sample["input"])
            for character in state.characters:
                unstructured = character.unstructured
                if self.preprocess_fn is not None:
                    unstructured = self.preprocess_fn(unstructured)
                # do some deduplication, if we expect different scores for the same description this may need to be changed
                if unstructured not in seen:
                    results.append(
                        {
                            "description": unstructured,
                            "kdma_value": getattr(character, key),
                        }
                    )
                    seen.add(unstructured)

        return results


def medical_preprocess(unstructured: str) -> str:
    return unstructured.split("\n")[0].strip()


def affiliation_preprocess(unstructured: str) -> str:
    return unstructured.split("\n")[-1].strip()


def merit_preprocess(unstructured: str) -> str:
    return unstructured.split("\n")[-1].strip()


def search_preprocess(unstructured: str) -> str:
    return unstructured.strip()


def personal_safety_preprocess(unstructured: str, situation: str = "") -> str:
    if situation:
        return f"{situation}\n{unstructured.strip()}"
    return unstructured.strip()


"""
datasets:
    medical: /data/shared/samba/phase2_icl/Feb2026-MU-train_20251218.json
    affiliation: /data/shared/samba/phase2_icl/Feb2026-AF-train_20251218.json
    merit: /data/shared/samba/phase2_icl/Feb2026-MF-train_20251218.json
    personal_safety: /data/shared/samba/phase2_icl/Feb2026-PS-train_20251218.json
    search: /data/shared/samba/phase2_icl/Feb2026-SS-train_20251218.json
"""
