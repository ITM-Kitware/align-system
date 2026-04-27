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
        extract_prompt,
        simplify_prompt,
        chunk_size=32,
        kdma_scale_factor=100,
        max_new_tokens=8192,
        prompt_header=None,
        prompt_footer=None,
        enable_caching=True,
    ):
        from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state

        self.state_hydration_fn = p2triage_hydrate_scenario_state

        self.kdma_name = kdma_name
        self.dataset = self._load_dataset(dataset)

        self.extract_prompt = extract_prompt
        self.simplify_prompt = simplify_prompt

        self.chunk_size = chunk_size
        self.kdma_scale_factor = kdma_scale_factor
        self.max_new_tokens = max_new_tokens

        self.prompt_header = prompt_header
        self.prompt_footer = prompt_footer

        self.enable_caching = enable_caching

        self.system_prompt = None

    def _extract(self, dataset, model):
        """
        Extract the relevant information from the dataset.
        """

        extract_prompt = OutlinesTemplate.from_string(self.extract_prompt)

        # Apply `self.extract_prompt` over the dataset in chunks. Accumulate the results.
        generator = outlines.Generator(model, KdmaScoreList)
        extracted = []
        for i in range(0, len(dataset), self.chunk_size):
            chunk = dataset[i : i + self.chunk_size]
            data = self._format_entries(chunk, scale=self.kdma_scale_factor)
            prompt = extract_prompt(dataset=data)
            result = generator(prompt, max_new_tokens=self.max_new_tokens)
            parsed = KdmaScoreList.model_validate_json(result).model_dump()["entries"]
            extracted.extend(parsed)

        log.debug(f"Extracted to {len(extracted)} entries in dataset.")

        return extracted

    def _simplify(self, dataset, model):
        """
        Simplify the information from the dataset.
        """

        simplify_prompt = OutlinesTemplate.from_string(self.simplify_prompt)

        generator = outlines.Generator(model, KdmaScoreList)

        data = self._format_entries(dataset, scale=1)
        prompt = simplify_prompt(dataset=data)
        result = generator(prompt, max_new_tokens=self.max_new_tokens)
        entries = KdmaScoreList.model_validate_json(result).model_dump()["entries"]

        log.debug(f"Simplified to {len(entries)} entries in dataset.")

        return entries

    @staticmethod
    def _deduplicate(dataset):
        """
        Simple deduplication of entries.
        """

        seen = {}
        for entry in dataset:
            if entry["description"] not in seen:
                seen[entry["description"]] = entry["kdma_value"]

        log.debug(f"Deduplicated to {len(seen)} entries in dataset.")

        return [{"description": k, "kdma_value": v} for k, v in seen.items()]

    def _build_prompt(self, dataset):
        """
        Build the system_prompt from dataset.
        """

        parts = [self.prompt_header] if self.prompt_header else []

        parts += [
            "\n".join(
                [
                    f"{x['description']} would score {int(x['kdma_value'])}"
                    for x in dataset
                ]
            )
        ]
        parts += [self.prompt_footer] if self.prompt_footer else []
        system_prompt = "\n".join(parts)

        return system_prompt

    def __call__(self, model):
        if self.system_prompt is not None:
            return self.system_prompt

        # Extract
        extract_cacher = ub.Cacher(
            "generated_system_prompt_template_extract",
            depends=self._cache_repr(self.dataset, self.extract_prompt),
            enabled=self.enable_caching,
        )
        extract_dataset = extract_cacher.tryload()
        if extract_dataset is None:
            if self.enable_caching:
                log.info("Cache miss for `generated_system_prompt_template_extract` ..")

            dataset = self._deduplicate(self.dataset)
            extract_dataset = self._extract(dataset, model)
            extract_cacher.save(extract_dataset)
        else:
            log.info("Cache hit for `generated_system_prompt_template_extract` ..")
        log.debug(f"Extract Dataset: {extract_dataset}")

        # Simplify
        simplify_cacher = ub.Cacher(
            "generated_system_prompt_template_extract_simplify",
            depends=self._cache_repr(extract_dataset, self.simplify_prompt),
            enabled=self.enable_caching,
        )
        simplify_dataset = simplify_cacher.tryload()
        if simplify_dataset is None:
            if self.enable_caching:
                log.info(
                    "Cache miss for `generated_system_prompt_template_extract_simplify` .."
                )

            dataset = self._deduplicate(extract_dataset)
            simplify_dataset = self._simplify(dataset, model)
            simplify_cacher.save(simplify_dataset)
        else:
            log.info(
                "Cache hit for `generated_system_prompt_template_extract_simplify` .."
            )
        log.debug(f"Simplify Dataset: {simplify_dataset}")

        self.system_prompt = self._build_prompt(simplify_dataset)

        return self.system_prompt

    @staticmethod
    def _format_entries(entries, scale=1.0):
        if len(entries) == 0:
            return ""

        return "\n".join(
            [
                f"RAW DESCRIPTION: {x['description'].replace(chr(10), ', ')} SCORE: {int(round(x['kdma_value'] * scale))}"
                for x in entries
            ]
        )

    def _load_dataset(self, dataset):
        key = "medical_condition" if self.kdma_name == "medical" else "attribute_rating"

        with open(dataset, mode="rb") as f:
            dataset = json.load(f)

        results = []
        for icl_sample in dataset:
            state, _ = self.state_hydration_fn(icl_sample["input"])
            for character in state.characters:
                kdma_value = getattr(character, key)
                if kdma_value == 0:
                    continue

                unstructured = character.unstructured
                results.append(
                    {
                        "description": unstructured,
                        "kdma_value": kdma_value,
                    }
                )

        log.debug(f"Loaded {len(results)} entries in dataset.")

        return results

    def _cache_repr(self, dataset, prompt):
        return f"{ub.hash_data(dataset)}-{self.kdma_name}-{self.chunk_size}-{self.kdma_scale_factor}-{self.max_new_tokens}-{ub.hash_data(prompt)}"


"""
datasets:
    medical: /data/shared/samba/phase2_icl/Feb2026-MU-train_20251218.json
    affiliation: /data/shared/samba/phase2_icl/Feb2026-AF-train_20251218.json
    merit: /data/shared/samba/phase2_icl/Feb2026-MF-train_20251218.json
    personal_safety: /data/shared/samba/phase2_icl/Feb2026-PS-train_20251218.json
    search: /data/shared/samba/phase2_icl/Feb2026-SS-train_20251218.json
"""
