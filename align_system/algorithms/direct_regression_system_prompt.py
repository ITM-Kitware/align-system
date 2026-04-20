import json
import logging
from typing import List

import outlines
import ubelt as ub
from outlines import Template as OutlinesTemplate
from pydantic import BaseModel

log = logging.getLogger(__name__)


class KdmaScore(BaseModel):
    description: str
    kdma_value: int


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
        # preprocess_fn=None,
        chunk_size=32,
        preprocess_prompt=None,
        enable_caching=True,
    ):
        from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state

        self.state_hydration_fn = p2triage_hydrate_scenario_state

        self.kdma_name = kdma_name

        self.template_prompt = OutlinesTemplate.from_string(template_prompt)

        self.kdma_scale_factor = kdma_scale_factor

        self.prompt_header = prompt_header
        self.prompt_footer = prompt_footer

        # self.preprocess_fn = preprocess_fn

        self.dataset = self._load_dataset(dataset)

        self.chunk_size = chunk_size

        self.system_prompt = None
        self.preprocess_prompt = (
            OutlinesTemplate.from_string(preprocess_prompt)
            if preprocess_prompt is not None
            else None
        )

        self.enable_caching = enable_caching

    def _extract_dataset(self, model, dataset):
        if self.preprocess_prompt is None:
            return dataset

        generator = outlines.Generator(model, KdmaScoreList)
        extracted = []
        for i in range(0, len(dataset), self.chunk_size):
            chunk = dataset[i : i + self.chunk_size]
            data = self._format_entries(chunk, scale=self.kdma_scale_factor)
            prompt = self.preprocess_prompt(dataset=data)
            # print(f'prompt length: {len(prompt)}')
            result = generator(prompt, max_new_tokens=8192)
            parsed = KdmaScoreList.model_validate_json(result).model_dump()["entries"]
            # print(f"Chunk {i // self.chunk_size}: {len(parsed)} entries extracted")
            # print(f"Sample: {parsed[:2]}")
            extracted.extend(parsed)

        seen = {}
        for entry in extracted:
            if entry["description"] not in seen:
                seen[entry["description"]] = entry["kdma_value"]

        return [{"description": k, "kdma_value": v} for k, v in seen.items()]

    def __call__(self, model):
        if self.system_prompt is not None:
            return self.system_prompt

        if self.enable_caching:
            cacher = ub.Cacher(
                "generated_system_prompt_template", self.cache_repr(), verbose=0
            )
            log.debug(f"cacher.fpath={cacher.fpath}")

            dataset = cacher.tryload()

            if dataset is not None:
                log.info("Cache hit for `generated_system_prompt_template`")
            else:
                log.info("Cache miss for `generated_system_prompt_template` ..")
                dataset = self._extract_dataset(model, self.dataset)
                cacher.save(dataset)
        else:
            dataset = self._extract_dataset(model, self.dataset)

        generator = outlines.Generator(model, KdmaScoreList)

        accumulated = []
        for i in range(0, len(dataset), self.chunk_size):
            chunk_data = self._format_entries(dataset[i : i + self.chunk_size], scale=1)
            # data = chunk_data
            data = "\n".join([self._format_entries(accumulated, scale=1), chunk_data])
            prompt = self.template_prompt(dataset=data)
            # print(f'prompt length: {len(prompt)}')
            result = generator(prompt, max_new_tokens=8192 * 2)
            # print(f'result length: {len(result)}')
            accumulated = KdmaScoreList.model_validate_json(result).model_dump()[
                "entries"
            ]
            # print(f"accumulated length {len(accumulated)}")
            # if len(accumulated) > 40:
            #     breakpoint()

        parts = [self.prompt_header] if self.prompt_header else []

        parts += [
            "\n".join(
                [
                    f"{x['description']} would score {int(x['kdma_value'])}"
                    for x in accumulated
                ]
            )
        ]
        parts += [self.prompt_footer] if self.prompt_footer else []
        self.system_prompt = "\n".join(parts)

        return self.system_prompt

    @staticmethod
    def _format_entries(entries, scale=1.0):
        if len(entries) == 0:
            return ""

        return "\n RAW DESCRIPTION: ".join(
            [
                f"{x['description'].replace(chr(10), ', ')} SCORE: {int(x['kdma_value'] * scale)}"
                for x in entries
            ]
        )

    def _load_dataset(self, dataset):
        key = "medical_condition" if self.kdma_name == "medical" else "attribute_rating"

        with open(dataset, mode="rb") as f:
            dataset = json.load(f)

        results = []
        seen = set()
        for icl_sample in dataset:
            state, actions = self.state_hydration_fn(icl_sample["input"])
            for character in state.characters:
                kdma_value = getattr(character, key)
                if kdma_value == 0:
                    continue

                unstructured = character.unstructured
                # do some deduplication, if we expect different scores for the same description this may need to be changed
                if unstructured not in seen:
                    results.append(
                        {
                            "description": unstructured,
                            "kdma_value": kdma_value,
                        }
                    )
                    seen.add(unstructured)

        return results

    # TODO
    def cache_repr(self):
        return None


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
