"""
DecisionFlow Prompt Engineering Module
========================================

This module contains prompt templates and schemas for the DecisionFlow pipeline.

Main modules:
- high_low_prompts: Binary high/low value prompts for all DecisionFlow stages
- fine_grained_prompts: Fine-grained value prompts with explicit scale anchors (Attribute + MathReason stages)
"""

from align_system.prompt_engineering.decision_flow.high_low_prompts import (
    # Variables Stage
    VariablesPrompt,
    VariablesOutputSchema,
    # Extraction Stage
    ExtractionPrompt,
    ExtractionOutputSchema,
    # Attribute Stage (High/Low)
    AttributePrompt,
    AttributeOutputSchema,
    # Filter Stage
    FilterPrompt,
    FilterOutputSchema,
    # Objective Stage
    ObjectivePrompt,
    ObjectiveOutputSchema,
    # Express Stage
    ExpressPrompt,
    ExpressOutputSchema,
    # MathReason Stage
    MathReasonPrompt,
    MathReasonOutputSchema,
)

from align_system.prompt_engineering.decision_flow.fine_grained_prompts import (
    # Phase 2 Fine-Grained Attribute Stage
    Phase2FineGrainedAttributePrompt,
    Phase2FineGrainedAttributeOutputSchema,
    # Fine-Grained MathReason Stage
    FineGrainedMathReasonPrompt,
    FineGrainedMathReasonOutputSchema,
)

__all__ = [
    # High/Low prompts
    'VariablesPrompt', 'VariablesOutputSchema',
    'ExtractionPrompt', 'ExtractionOutputSchema',
    'AttributePrompt', 'AttributeOutputSchema',
    'FilterPrompt', 'FilterOutputSchema',
    'ObjectivePrompt', 'ObjectiveOutputSchema',
    'ExpressPrompt', 'ExpressOutputSchema',
    'MathReasonPrompt', 'MathReasonOutputSchema',
    # Fine-grained prompts
    'Phase2FineGrainedAttributePrompt', 'Phase2FineGrainedAttributeOutputSchema',
    'FineGrainedMathReasonPrompt', 'FineGrainedMathReasonOutputSchema',
]
