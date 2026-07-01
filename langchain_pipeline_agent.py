#!/usr/bin/env python
"""LangChain agent that designs and executes an ALIGN-system pipeline.

This is a standalone proof-of-concept that wires the project's existing
``PipelineADM`` / ``ADMComponent`` building blocks up to a LangChain agent.
Given a triage scenario, the agent reasons about *which* components to chain
together (and in what order), assembles a ``PipelineADM`` from them, and runs
it to choose an action.

Everything runs against **local LLMs via Ollama** -- no Anthropic, Gemini, or
OpenAI calls are made. Two local models are used:

  * an *agent* model (default ``llama3.1:latest``) that does the tool-calling /
    pipeline-design reasoning, driven through ``langchain_ollama.ChatOllama``;
  * a *decision* model (default ``gemma4:latest``) used inside the pipeline's
    LLM decision component, driven through the project's own
    ``OllamaInferenceEngine``.

Usage
-----
    ./run_pipeline_agent.sh --scenario-index 0

NOTE on environments: this agent needs LangChain >= 1.0 (for ``create_agent``).
The conda ``align`` env is pinned to LangChain 0.0.308 by the old
``llama_index 0.8.42`` dependency, so it CANNOT run there. Use
``run_pipeline_agent.sh`` (which invokes the pyenv ``pyenv3.11.2`` interpreter
that has LangChain 1.x + ``align_system`` installed), or any env that has both
``langchain>=1.0`` / ``langchain-ollama`` and ``align_system`` importable.

Also requires a running Ollama server (``ollama serve``) with the chosen
models pulled.
"""

import argparse
import json
import os
import textwrap

from omegaconf import OmegaConf
from hydra.utils import instantiate

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from align_system.algorithms.abstracts import ADMComponent
from align_system.algorithms.ollama_inference_engine import OllamaInferenceEngine
from align_system.algorithms.pipeline_adm import PipelineADM
from align_system.algorithms.misc_itm_adm_components import (
    EnsureChosenActionADMComponent,
    ITMFormatChoicesADMComponent,
    JustificationFromReasonings,
    Phase2RegressionRuleBasedCorrection,
    PopulateChoiceInfo,
)
from align_system.algorithms.comparative_regression_adm_component import (
    ComparativeRegressionADMComponent,
    RegressionOutputsConflictResolver,
)
from align_system.algorithms.alignment_adm_component import (
    AlignmentADMComponent,
    RandomEffectsModelAlignmentADMComponent,
)
from align_system.algorithms.choose_idx_adm_component import ChooseIdxADMComponent
from align_system.algorithms.random_adm_component import RandomChoiceADMComponent
from align_system.utils.alignment_utils import AvgDistScalarAlignment
from align_system.prompt_engineering.outlines_prompts import (
    ComparativeRegressionSystemPrompt,
)
from align_system.interfaces.input_output_file import InputOutputFileInterface

# Directory holding the project's Hydra config tree (attributes / templates).
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "align_system", "configs")

# Phase2 attributes available for comparative regression: kdma -> config name.
PHASE2_ATTRIBUTES = {
    "medical": "medical_urgency",
    "affiliation": "affiliation_focus",
    "merit": "merit_focus",
    "search": "search_or_stay",
    "personal_safety": "personal_safety",
}


# Default location of the few-shot ICL training data (Feb2026 datasets).
ICL_DATA_DIR = "/data/shared/samba/phase2_icl"
ICL_DATASETS = {
    "medical": "Feb2026-MU-train_20251218.json",
    "affiliation": "Feb2026-AF-train_20251218.json",
    "merit": "Feb2026-MF-train_20251218.json",
    "personal_safety": "Feb2026-PS-train_20251218.json",
    "search": "Feb2026-SS-train_20251218.json",
}


class _HashableDict(dict):
    """A dict usable as an lru_cache key (identity-hashed). The ICL component's
    ``init_icl_engine_from_target`` is ``@lru_cache``'d and receives the
    attributes mapping, which a plain dict can't satisfy."""
    def __hash__(self):
        return id(self)


def _load_cfg(*parts):
    return instantiate(OmegaConf.load(os.path.join(_CONFIG_DIR, *parts)))


def build_phase2_regression_kit():
    """Instantiate the attributes + prompt/schema templates used by the
    Phase2 comparative-regression pipeline (from the project's configs)."""
    attributes = _HashableDict({
        kdma: _load_cfg("attribute", f"{name}.yaml")
        for kdma, name in PHASE2_ATTRIBUTES.items()
    })
    templates = {
        "scenario_description": _load_cfg(
            "template", "scenario_description", "phase2.yaml"),
        "prompt": _load_cfg(
            "template", "prompt", "phase2_comparative_regression.yaml"),
        "score_schema": _load_cfg(
            "template", "output_schema", "phase2_comparative_regression_choice.yaml"),
        "system_prompt": ComparativeRegressionSystemPrompt(),
    }
    return attributes, templates


def load_alignment_target(name):
    """Load an alignment-target config (e.g. 'feb2026/Feb2026-MF-1') as a plain
    dict. Phase2 targets are random-effects models: value is null and the
    behaviour is set by `parameters` (intercept / medical_weight / attr_weight)."""
    path = os.path.join(_CONFIG_DIR, "alignment_target", f"{name}.yaml")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def target_is_random_effects(alignment_target):
    """True if any KDMA in the target is parameterised (random-effects model)
    rather than a plain scalar value."""
    if alignment_target is None:
        return False
    for kv in alignment_target["kdma_values"]:
        if kv.get("parameters") is not None and kv.get("value") is None:
            return True
    return False


def enable_gpu_bert_similarity(device="auto"):
    """Put the ICL BERT-similarity scorer on the GPU (contained monkeypatch).

    align_system hardcodes ``BERTScorer(device="cpu")`` to keep VRAM free for
    the generative model on cluster runs, which makes ICL example embedding
    very slow locally. roberta-large is only ~1.4GB and fits alongside the
    Ollama model on an 8GB GPU, so we swap in a GPU scorer just for this script
    (we do NOT modify the shared project code)."""
    from align_system.utils import incontext_utils as _icu
    from bert_score import BERTScorer

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _gpu_get_bert_scorer(bert_model_type=None):
        model_type = (_icu.BERT_MODEL_ALIASES.get(bert_model_type, bert_model_type)
                      or "roberta-large")
        return BERTScorer(model_type=model_type, device=device)

    _icu._get_bert_scorer = _gpu_get_bert_scorer
    return device


def build_icl_component(attributes, templates, icl_data_dir, num_examples=1):
    """Build the few-shot ICL component, pointing at local training datasets.

    Mirrors configs/adm_component/icl/phase2_comparative.yaml but overrides the
    dataset paths to ``icl_data_dir`` (the Feb2026 train files) and instantiates
    the generator partial directly (the Hydra config injects templates/attributes
    separately via the ADM config)."""
    from align_system.algorithms.icl_adm_component import ICLADMComponent

    cfg = OmegaConf.load(os.path.join(
        _CONFIG_DIR, "adm_component", "icl", "phase2_comparative.yaml"))
    gen_node = cfg.icl_generator_partial
    gen_node.incontext_settings.datasets = {
        kdma: os.path.join(icl_data_dir, fname)
        for kdma, fname in ICL_DATASETS.items()
    }
    gen_node.incontext_settings.number = num_examples
    gen_node.incontext_settings.leave_one_out_strategy = None

    return ICLADMComponent(
        icl_generator_partial=instantiate(gen_node),
        scenario_description_template=templates["scenario_description"],
        prompt_template=templates["prompt"],
        attributes=attributes,
        # Must match comparative_regression's override so ICL produces examples
        # for every attribute the regression step scores (medical + target);
        # otherwise the regression loop KeyErrors on the missing 'medical' key.
        target_attribute_names_override=["medical", "*"],
    )


# ---------------------------------------------------------------------------
# A small custom ADM component: make the triage decision with a LOCAL LLM.
# It reuses the project's OllamaInferenceEngine for structured JSON output.
# ---------------------------------------------------------------------------
class OllamaLLMChoiceADMComponent(ADMComponent):
    """Pick the best choice using a local Ollama model (structured output)."""

    def __init__(self, structured_inference_engine):
        self.structured_inference_engine = structured_inference_engine

    def run_returns(self):
        return ("chosen_choice", "justification")

    def run(self, scenario_state, choices):
        situation = getattr(scenario_state, "unstructured", "") or ""
        numbered = "\n".join(f"  [{i}] {c}" for i, c in enumerate(choices))

        system_prompt = (
            "You are an expert battlefield medical triage decision maker. "
            "Given a situation and a list of possible actions, select the single "
            "best action. Respond ONLY with JSON."
        )
        user_prompt = (
            f"Situation:\n{situation.strip()}\n\n"
            f"Possible actions:\n{numbered}\n\n"
            "Choose the index of the single best action and briefly justify it."
        )
        dialog = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "chosen_index": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": len(choices) - 1,
                    },
                },
                "required": ["reasoning", "chosen_index"],
            }
        )

        prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        result = self.structured_inference_engine.run_inference(prompt, schema)[0]

        idx = int(result.get("chosen_index", 0)) % len(choices)
        justification = result.get("reasoning", "") or "Chosen by local LLM."
        return choices[idx], justification


# ---------------------------------------------------------------------------
# Engines are built lazily from the model names currently on SESSION, so the
# agent can switch which local Ollama model a step uses (see set_pipeline_models).
# ---------------------------------------------------------------------------
def make_ollama_engine(model_name, host="http://localhost:11434", num_ctx=None):
    return OllamaInferenceEngine(
        model_name=model_name, host=host, temperature=0.0, num_ctx=num_ctx)


def _decision_engine():
    return make_ollama_engine(SESSION.decision_model, SESSION.ollama_host)


def _regression_engine():
    return make_ollama_engine(
        SESSION.regression_model, SESSION.ollama_host, SESSION.regression_num_ctx)


# ---------------------------------------------------------------------------
# Registry of pipeline building blocks the agent is allowed to assemble.
# Each entry is (factory, "produces ...", "requires ...", human description).
# ---------------------------------------------------------------------------
def build_component_registry(attributes, templates, icl_factory=None):
    registry = {
        "format_choices": {
            "factory": lambda: ITMFormatChoicesADMComponent(),
            "produces": ["choices"],
            "requires": ["scenario_state", "actions"],
            "description": "Format the available actions into human-readable "
            "choice strings. Almost every pipeline needs this first.",
        },
        # ---- few-shot in-context-learning (optional, before regression) ----
        **({"regression_icl": {
            "factory": icl_factory,
            "produces": ["icl_dialog_elements", "icl_example_info"],
            "requires": ["scenario_state", "choices", "actions",
                         "alignment_target"],
            "description": "Retrieve few-shot in-context examples (BERT "
            "similarity) from local training data for the target attribute(s). "
            "Optional; place between format_choices and comparative_regression "
            "to improve KDMA score calibration.",
        }} if icl_factory is not None else {}),
        # ---- comparative-regression family (KDMA scoring + alignment) ----
        "comparative_regression": {
            "factory": lambda: ComparativeRegressionADMComponent(
                _regression_engine(),
                templates["scenario_description"],
                templates["prompt"],
                templates["score_schema"],
                attributes=attributes,
                system_prompt_template=templates["system_prompt"],
                # Always score 'medical' plus the target attribute(s): the
                # random-effects alignment needs medical-urgency for each choice
                # (matches phase2_comparative_no_template).
                target_attribute_names_override=["medical", "*"],
                output_conflict_resolver=RegressionOutputsConflictResolver(),
            ),
            "produces": ["attribute_prediction_reasonings",
                         "attribute_prediction_scores", "attribute_dialogs"],
            "requires": ["scenario_state", "choices", "alignment_target"],
            "description": "Use a LOCAL LLM to predict, for every choice, a "
            "score for medical-urgency plus each attribute (KDMA) in the "
            "alignment target. Produces per-choice scores + reasonings. Needs "
            "an alignment target.",
        },
        "regression_rule_based_correction": {
            "factory": lambda: Phase2RegressionRuleBasedCorrection(),
            "produces": ["attribute_prediction_scores"],
            "requires": ["attribute_prediction_scores"],
            "description": "Apply Phase2 hand-written rules to fix up predicted "
            "KDMA scores (optional, runs after comparative_regression).",
        },
        "scalar_alignment": {
            "factory": lambda: AlignmentADMComponent(
                AvgDistScalarAlignment(), attributes=attributes),
            "produces": ["chosen_choice", "best_sample_idx"],
            "requires": ["scenario_state", "attribute_prediction_scores",
                         "alignment_target"],
            "description": "Pick the choice whose predicted scores are closest "
            "to the scalar alignment target (avg-distance). Use this decision "
            "step ONLY for a SCALAR alignment target.",
        },
        "random_effects_alignment": {
            "factory": lambda: RandomEffectsModelAlignmentADMComponent(
                attributes=attributes),
            "produces": ["chosen_choice", "best_sample_idx", "alignment_info"],
            "requires": ["attribute_prediction_scores", "alignment_target"],
            "description": "Select between exactly TWO choices using ADEPT's "
            "random-effects logistic model (intercept + medical/attr weights). "
            "Use this decision step for a RANDOM-EFFECTS alignment target (the "
            "Phase2 Feb targets). Requires medical-urgency scores.",
        },
        "justification_from_reasonings": {
            "factory": lambda: JustificationFromReasonings(),
            "produces": ["justification"],
            "requires": ["attribute_prediction_reasonings", "chosen_choice",
                         "best_sample_idx"],
            "description": "Build a justification string from the regression "
            "reasonings for the chosen choice (use after scalar_alignment).",
        },
        "llm_choice": {
            "factory": lambda: OllamaLLMChoiceADMComponent(_decision_engine()),
            "produces": ["chosen_choice", "justification"],
            "requires": ["scenario_state", "choices"],
            "description": "Use a LOCAL Ollama LLM to reason over the situation "
            "and pick the best choice with a justification.",
        },
        "random_choice": {
            "factory": lambda: RandomChoiceADMComponent(),
            "produces": ["chosen_choice", "justification"],
            "requires": ["choices"],
            "description": "Pick a choice uniformly at random (baseline).",
        },
        "first_choice": {
            "factory": lambda: ChooseIdxADMComponent(choice_idx=0),
            "produces": ["chosen_choice", "justification"],
            "requires": ["choices"],
            "description": "Always pick the first choice (deterministic baseline).",
        },
        "ensure_chosen_action": {
            "factory": lambda: EnsureChosenActionADMComponent(),
            "produces": ["chosen_action"],
            "requires": ["choices", "actions", "chosen_choice"],
            "description": "Resolve the selected choice back into a concrete "
            "action object. Required to actually return an action.",
        },
        "populate_choice_info": {
            "factory": lambda: PopulateChoiceInfo(),
            "produces": ["choice_info"],
            "requires": ["choices", "actions"],
            "description": "Attach bookkeeping/metadata about the choices "
            "(optional, but nice for inspection).",
        },
    }
    return registry


# ---------------------------------------------------------------------------
# Session state shared with the agent's tools.
# ---------------------------------------------------------------------------
# KDMAs the agent can align to, and the Feb target-config prefix for each.
# (medical urgency is always scored as a modifier but is not an align target.)
ALIGNABLE_KDMA_TO_PREFIX = {
    "affiliation": "AF",
    "merit": "MF",
    "personal_safety": "PS",
    "search": "SS",
}


def list_feb_target_variants(kdma):
    """Variant numbers available for an alignable KDMA's Feb targets."""
    prefix = ALIGNABLE_KDMA_TO_PREFIX[kdma]
    d = os.path.join(_CONFIG_DIR, "alignment_target", "feb2026")
    import re as _re
    variants = []
    for fn in os.listdir(d):
        mobj = _re.fullmatch(rf"Feb2026-{prefix}-(\d+)\.yaml", fn)
        if mobj:
            variants.append(int(mobj.group(1)))
    return sorted(variants)


class Session:
    def __init__(self, registry, scenario_state, actions, scenario_id,
                 alignment_target=None, attributes=None,
                 decision_model="gemma4:latest", regression_model="gemma4:latest",
                 regression_num_ctx=16384, ollama_host="http://localhost:11434"):
        self.registry = registry
        self.scenario_state = scenario_state
        self.actions = actions
        self.scenario_id = scenario_id
        self.alignment_target = alignment_target
        self.attributes = attributes or {}
        # Models the agent can change at runtime via set_pipeline_models.
        self.decision_model = decision_model
        self.regression_model = regression_model
        self.regression_num_ctx = regression_num_ctx
        self.ollama_host = ollama_host


SESSION: Session = None  # populated in main()


def make_tools():
    @tool
    def describe_scenario() -> str:
        """Return the loaded triage scenario: its situation text and the
        available actions to choose from."""
        choices = [a.unstructured for a in SESSION.actions]
        numbered = "\n".join(f"  [{i}] {c}" for i, c in enumerate(choices))
        situation = getattr(SESSION.scenario_state, "unstructured", "") or ""
        if SESSION.alignment_target is None:
            target_str = "none (no attribute to align to)"
        elif target_is_random_effects(SESSION.alignment_target):
            kdmas = ", ".join(k["kdma"] for k in SESSION.alignment_target["kdma_values"])
            target_str = (f"RANDOM-EFFECTS model on [{kdmas}] "
                          "(use random_effects_alignment; needs exactly 2 choices)")
        else:
            kvs = SESSION.alignment_target["kdma_values"]
            target_str = ("SCALAR " + ", ".join(f"{k['kdma']}={k['value']}" for k in kvs)
                          + " (use scalar_alignment)")
        return (
            f"Scenario id: {SESSION.scenario_id}\n"
            f"Situation:\n{situation.strip()}\n\n"
            f"Available actions ({len(choices)}):\n{numbered}\n\n"
            f"Alignment target: {target_str}"
        )

    @tool
    def analyze_relevant_attributes() -> str:
        """Analyze the scenario with a local LLM to judge which alignment
        attribute (KDMA) is most relevant to the decision. Returns a per-KDMA
        relevance score (0-1) with reasoning and the single most-relevant KDMA.
        Use this to decide what to align to when no target is preset."""
        situation = getattr(SESSION.scenario_state, "unstructured", "") or ""
        choices = [a.unstructured for a in SESSION.actions]
        attr_lines = []
        for kdma in ALIGNABLE_KDMA_TO_PREFIX:
            attr = SESSION.attributes[kdma]
            attr_lines.append(f"- {kdma} ({attr.name}): {attr.description}")
        attrs_block = "\n".join(attr_lines)
        names = list(ALIGNABLE_KDMA_TO_PREFIX)

        system_prompt = (
            "You are an expert in medical-triage value alignment. Given a "
            "scenario and a set of candidate decision-making attributes, judge "
            "how relevant each attribute is to THIS decision. Respond ONLY JSON."
        )
        user_prompt = (
            f"Situation:\n{situation.strip()}\n\n"
            f"Choices: {choices}\n\n"
            f"Candidate attributes:\n{attrs_block}\n\n"
            "For each attribute give a relevance score in [0,1] and a one-sentence "
            "reason, then name the single most relevant attribute."
        )
        schema = json.dumps({
            "type": "object",
            "properties": {
                "relevance": {
                    "type": "object",
                    "properties": {
                        k: {"type": "object", "properties": {
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning": {"type": "string"}},
                            "required": ["score", "reasoning"]}
                        for k in names
                    },
                    "required": names,
                },
                "most_relevant": {"type": "string", "enum": names},
            },
            "required": ["relevance", "most_relevant"],
        })
        dialog = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]
        engine = _regression_engine()
        prompt = engine.dialog_to_prompt(dialog)
        result = engine.run_inference(prompt, schema)[0]
        return json.dumps(result, indent=2)

    @tool
    def list_local_models() -> str:
        """List the local Ollama models available to use for the pipeline's LLM
        steps (decision / KDMA-regression scoring). Use before set_pipeline_models."""
        import ollama
        resp = ollama.Client(host=SESSION.ollama_host).list()
        lines = []
        for mdl in resp.models:
            gb = (mdl.size or 0) / 1e9
            lines.append(f"- {mdl.model}  (~{gb:.1f} GB)")
        return ("Local Ollama models (current: decision="
                f"{SESSION.decision_model}, regression={SESSION.regression_model}):\n"
                + "\n".join(sorted(lines)))

    @tool
    def set_pipeline_models(decision_model: str = None,
                            regression_model: str = None) -> str:
        """Choose which local Ollama model the pipeline uses for its LLM steps.
        - decision_model: used by the 'llm_choice' step.
        - regression_model: used by 'comparative_regression' (KDMA scoring) and
          by scenario analysis.
        Pass a model name from list_local_models. Either or both may be set."""
        import ollama
        available = {m.model for m in ollama.Client(host=SESSION.ollama_host).list().models}
        for label, val in (("decision_model", decision_model),
                           ("regression_model", regression_model)):
            if val is not None and val not in available:
                return (f"ERROR: model '{val}' for {label} is not available. "
                        "Use list_local_models to see valid names.")
        if decision_model is not None:
            SESSION.decision_model = decision_model
        if regression_model is not None:
            SESSION.regression_model = regression_model
        return (f"Pipeline models set: decision={SESSION.decision_model}, "
                f"regression={SESSION.regression_model}.")

    @tool
    def list_alignment_targets() -> str:
        """List the alignment targets the agent may choose from. Each alignable
        KDMA has several Feb parameter variants; pass a full name like
        'feb2026/Feb2026-MF-1' to set_alignment_target."""
        lines = []
        for kdma, prefix in ALIGNABLE_KDMA_TO_PREFIX.items():
            variants = list_feb_target_variants(kdma)
            lines.append(
                f"- {kdma}: feb2026/Feb2026-{prefix}-<n>  (n in {variants}); "
                f"variants are different alignment preference levels (default -1)")
        return ("Alignable KDMAs and their Feb random-effects targets:\n"
                + "\n".join(lines))

    @tool
    def set_alignment_target(name: str) -> str:
        """Set the scenario's alignment target by config name, e.g.
        'feb2026/Feb2026-MF-1'. This decides which KDMA the pipeline aligns to.
        Call this (typically after analyze_relevant_attributes) before building
        a comparative-regression pipeline."""
        path = os.path.join(_CONFIG_DIR, "alignment_target", f"{name}.yaml")
        if not os.path.isfile(path):
            return (f"ERROR: no such alignment target '{name}'. Use "
                    "list_alignment_targets to see valid names.")
        SESSION.alignment_target = load_alignment_target(name)
        kdmas = ", ".join(k["kdma"] for k in SESSION.alignment_target["kdma_values"])
        kind = ("random-effects" if target_is_random_effects(SESSION.alignment_target)
                else "scalar")
        return (f"Alignment target set to '{name}' ({kind} on [{kdmas}]). "
                "Use random_effects_alignment for random-effects targets, "
                "scalar_alignment for scalar.")

    @tool
    def list_available_components() -> str:
        """List the pipeline building-block components that can be assembled
        into a pipeline, including what each produces and requires."""
        lines = []
        for name, info in SESSION.registry.items():
            lines.append(
                f"- {name}: {info['description']}\n"
                f"    requires: {info['requires']}\n"
                f"    produces: {info['produces']}"
            )
        return "\n".join(lines)

    @tool
    def build_and_run_pipeline(steps: list[str]) -> str:
        """Build a pipeline from an ordered list of component names and execute
        it on the loaded scenario. Returns the chosen action and justification.

        `steps` must be a list of component names (see list_available_components),
        in execution order. A valid pipeline must produce a `chosen_action`
        (typically: format_choices -> a decision step -> ensure_chosen_action).
        """
        registry = SESSION.registry
        unknown = [s for s in steps if s not in registry]
        if unknown:
            return f"ERROR: unknown component(s): {unknown}. Valid: {list(registry)}"

        # Validate data-flow: each component's required inputs must be available.
        available = {"scenario_state", "actions"}
        if SESSION.alignment_target is not None:
            available.add("alignment_target")
        for s in steps:
            missing = [r for r in registry[s]["requires"] if r not in available]
            if missing:
                if "alignment_target" in missing and SESSION.alignment_target is None:
                    return (
                        f"ERROR: component '{s}' needs an alignment target, but "
                        "none was set. Re-run with --align-attribute/--align-value "
                        "to use the comparative-regression pipeline, or build a "
                        "simple pipeline (e.g. llm_choice) instead."
                    )
                return (
                    f"ERROR: component '{s}' requires {missing} which are not "
                    f"produced by an earlier step. Reorder or add steps."
                )
            available.update(registry[s]["produces"])

        if "chosen_action" not in available:
            return (
                "ERROR: this pipeline never produces a 'chosen_action'. Add a "
                "decision step (e.g. llm_choice) and 'ensure_chosen_action'."
            )

        components = [registry[s]["factory"]() for s in steps]
        pipeline = PipelineADM(steps=components)

        chosen_action, working_output = pipeline.choose_action(
            scenario_state=SESSION.scenario_state,
            available_actions=SESSION.actions,
            alignment_target=SESSION.alignment_target,
        )

        result = {
            "pipeline": steps,
            "chosen_action": getattr(chosen_action, "unstructured", str(chosen_action)),
            "action_type": getattr(chosen_action, "action_type", None),
            "justification": working_output.get("justification", ""),
        }
        if "attribute_prediction_scores" in working_output:
            result["predicted_kdma_scores"] = working_output["attribute_prediction_scores"]
        return json.dumps(result, indent=2, default=str)

    return [describe_scenario, list_local_models, set_pipeline_models,
            analyze_relevant_attributes,
            list_alignment_targets, set_alignment_target,
            list_available_components, build_and_run_pipeline]


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous decision-making agent for the ALIGN triage system.
    Your job is to DESIGN a decision pipeline and EXECUTE it on a scenario.

    Work in this order:
      1. Call describe_scenario to understand the situation, the choices, and
         whether an alignment target is already set.
      1b. CHOOSE WHICH LOCAL MODELS TO USE (optional): call list_local_models
         to see what's installed, then set_pipeline_models to pick the model for
         the decision step and/or the KDMA-regression scoring step. Prefer a
         capable instruct model that fits in VRAM; a larger model gives better
         KDMA calibration but is slower. If unsure, keep the defaults.
      2. CHOOSE WHAT TO ALIGN TO (if no target is already set):
         - Call analyze_relevant_attributes to judge, from the scenario, which
           attribute (KDMA) most drives this decision.
         - Call list_alignment_targets to see the available targets.
         - Call set_alignment_target with the target whose KDMA matches your
           analysis (e.g. the most-relevant KDMA's '-1' variant). Briefly
           explain WHY that attribute fits the scenario.
         If a target is already set, you may skip this and use it as-is.
      3. Call list_available_components to see the building blocks you can use.
      4. Decide on an ordered list of components that forms a valid pipeline.
         Every valid pipeline starts by formatting choices, then makes a
         decision, then resolves that decision into a concrete action
         (ensure_chosen_action). There are two main archetypes:

         (a) Simple LLM-choice pipeline (use only when you deliberately do NOT
             align -- no target set):
             format_choices -> llm_choice -> ensure_chosen_action
             [-> populate_choice_info]

         (b) Comparative-regression pipeline (use when an alignment target IS
             set -- it picks the action whose predicted attribute/KDMA scores
             best match the target):
             format_choices [-> regression_icl] -> comparative_regression
             [-> regression_rule_based_correction] -> <alignment>
             -> justification_from_reasonings -> ensure_chosen_action
             [-> populate_choice_info]
             Include regression_icl (if available) to add few-shot examples
             that calibrate the KDMA scoring.
             Pick <alignment> based on the target type reported by
             describe_scenario / set_alignment_target: use scalar_alignment for
             a SCALAR target, or random_effects_alignment for a RANDOM-EFFECTS
             target.
      5. Call build_and_run_pipeline with your ordered list of step names.
      6. Report your analysis, the alignment target you chose and why, the
         pipeline you built, the action it chose, and the reasoning.

    Use ONLY the provided tools. Do not invent component or target names.
    """
).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-file",
        default="example_data/input_output_files/oracle_adept_training_input_output.json",
        help="Path to an ALIGN input/output JSON file.",
    )
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument(
        "--state-hydration-domain",
        default="p1",
        choices=["p1", "p2triage", "minimal"],
    )
    parser.add_argument("--agent-model", default="qwen2.5:32b",
                        help="Local Ollama model for the agent (must support tools). "
                             "qwen2.5 is far more reliable than llama3.1 for the "
                             "multi-step tool-calling this agent does.")
    parser.add_argument("--decision-model", default="gemma4:latest",
                        help="Local Ollama model for the in-pipeline LLM decision step.")
    parser.add_argument("--regression-model", default="gemma4:latest",
                        help="Local Ollama model for the comparative-regression "
                             "KDMA scoring step.")
    parser.add_argument("--align-attribute", default=None,
                        choices=list(PHASE2_ATTRIBUTES),
                        help="If set, build an alignment target for this KDMA, "
                             "enabling the comparative-regression pipeline.")
    parser.add_argument("--align-value", type=float, default=0.9,
                        help="Target value (0-1) for --align-attribute.")
    parser.add_argument("--icl-data-dir", default=ICL_DATA_DIR,
                        help="Directory with the few-shot ICL training datasets. "
                             "Registers the optional 'regression_icl' component "
                             "if present.")
    parser.add_argument("--icl-num-examples", type=int, default=20,
                        help="Number of few-shot examples per attribute.")
    parser.add_argument("--num-ctx", type=int, default=16384,
                        help="Ollama context window for the regression engine "
                             "(few-shot prompts are large).")
    parser.add_argument("--bert-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for the ICL BERT-similarity embedding. "
                             "'auto' uses the GPU if available (much faster than "
                             "the project default of CPU).")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    # ---- interface: local file (default) or live TA3 service ------------
    parser.add_argument("--interface", default="file", choices=["file", "ta3"],
                        help="Where scenarios come from: a local input/output "
                             "file, or the live TA3 eval service.")
    parser.add_argument("--api-endpoint", default="http://127.0.0.1:8080",
                        help="TA3 service endpoint (--interface ta3).")
    parser.add_argument("--scenario-id", default=None,
                        help="TA3 scenario id to run, e.g. Feb2026-MF0-train.")
    parser.add_argument("--session-type", default="adept",
                        help="TA3 session type (--interface ta3).")
    parser.add_argument("--ta3-domain", default="p2triage",
                        help="TA3 domain (--interface ta3).")
    parser.add_argument("--training-session", default="full",
                        choices=["full", "solo"],
                        help="TA3 kdma_training mode (--interface ta3).")
    # ---- alignment target from a config (overrides --align-attribute) ----
    parser.add_argument("--alignment-target", default=None,
                        help="Alignment-target config name, e.g. "
                             "'feb2026/Feb2026-MF-1'. Loaded from "
                             "align_system/configs/alignment_target/. Overrides "
                             "--align-attribute and enables random-effects "
                             "alignment when the target is parameterised.")
    args = parser.parse_args()

    # ---- load a scenario -------------------------------------------------
    if args.interface == "ta3":
        from align_system.interfaces.ta3_caci_action_based_service import (
            TA3CACIActionBasedServiceInterface)
        if args.scenario_id is None:
            raise SystemExit("--scenario-id is required with --interface ta3")
        iface = TA3CACIActionBasedServiceInterface(
            username="ALIGN-AGENT",
            api_endpoint=args.api_endpoint,
            session_type=args.session_type,
            scenario_ids=[args.scenario_id],
            domain=args.ta3_domain,
            training_session=args.training_session,
        )
        scenario = iface.start_scenario()
        if scenario is None:
            raise SystemExit(f"TA3 returned no scenario for {args.scenario_id!r}")
    else:
        iface = InputOutputFileInterface(
            args.scenario_file, state_hydration_domain=args.state_hydration_domain
        )
        scenario = None
        for _ in range(args.scenario_index + 1):
            scenario = iface.start_scenario()
            if scenario is None:
                raise SystemExit(f"No scenario at index {args.scenario_index}")

    scenario_state = scenario.get_state()
    actions = scenario.get_available_actions()

    # ---- component registry (engines are built lazily from SESSION's
    #      current model choice, which the agent may change at runtime) -----
    attributes, templates = build_phase2_regression_kit()

    # Register the optional few-shot ICL component only if its data is present.
    icl_factory = None
    if os.path.isdir(args.icl_data_dir):
        bert_device = enable_gpu_bert_similarity(args.bert_device)
        print(f"(ICL BERT-similarity embedding device: {bert_device})")
        icl_factory = lambda: build_icl_component(  # noqa: E731
            attributes, templates, args.icl_data_dir, args.icl_num_examples)
    else:
        print(f"(note: ICL data dir {args.icl_data_dir!r} not found; "
              "'regression_icl' component disabled)")

    registry = build_component_registry(
        attributes, templates, icl_factory=icl_factory)

    # ---- optional alignment target (enables comparative regression) -----
    alignment_target = None
    if args.alignment_target is not None:
        alignment_target = load_alignment_target(args.alignment_target)
    elif args.align_attribute is not None:
        alignment_target = {
            "kdma_values": [
                {"kdma": args.align_attribute, "value": args.align_value}
            ]
        }

    global SESSION
    SESSION = Session(registry, scenario_state, actions, scenario.id(),
                      alignment_target=alignment_target, attributes=attributes,
                      decision_model=args.decision_model,
                      regression_model=args.regression_model,
                      regression_num_ctx=args.num_ctx,
                      ollama_host=args.ollama_host)

    # ---- the LangChain agent, backed by a LOCAL model -------------------
    llm = ChatOllama(model=args.agent_model, temperature=0.0, base_url=args.ollama_host)
    agent = create_agent(llm, tools=make_tools(), system_prompt=SYSTEM_PROMPT)

    if alignment_target is None:
        user_msg = (
            "No alignment target is preset. First analyze the scenario to decide "
            "which attribute (KDMA) is most relevant, choose and set a matching "
            "alignment target, then design and run a comparative-regression "
            "pipeline that selects the action best aligned to it. Summarize your "
            "analysis, the target you chose and why, the pipeline, and the action."
        )
    else:
        kind = ("random-effects" if target_is_random_effects(alignment_target)
                else "scalar")
        user_msg = (
            f"A {kind} alignment target is set for this scenario. Design and run "
            "a comparative-regression pipeline that predicts attribute/KDMA "
            "scores and selects the action best aligned to the target (choose the "
            "alignment step that matches the target type). Then summarize the "
            "pipeline you built and the action it selected."
        )

    align_label = (args.alignment_target or
                   (f"{args.align_attribute}={args.align_value}"
                    if args.align_attribute else "agent-chooses"))
    print(f"\n=== Scenario: {scenario.id()} | interface={args.interface} | "
          f"agent={args.agent_model} | regression={args.regression_model} | "
          f"align={align_label} ===\n")

    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})

    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"[agent -> tool] {tc['name']}({tc['args']})")
        content = (msg.content or "").strip() if isinstance(msg.content, str) else msg.content
        if content:
            print(f"[{role}] {content}\n")


if __name__ == "__main__":
    main()
