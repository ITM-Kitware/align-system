# CAGE 2 Environment Documentation

## LLM agent defender

To run the LLM defender use this:
```
python './align-system/align_system/cli/LLM_defender_CAGE2.py' +experiment=phase2_july_collab/pipeline_baseline_single_cage2
```

## Red attacker classification by LLM agent

To run the red attacker classification by a LLM use this:
```
python './align-system/align_system/cli/red_agent_classification.py' +experiment=phase2_july_collab/pipeline_baseline_red_agent_classification
```

## Hybrid agent (LLM + RL)

To run hybrid framework, where a LLM predicts the type of attacker from a set of observations and then a specialized RL agent is selected, use this:
```
python './align-system/align_system/cli/Test_single_agent.py' +experiment=phase2_july_collab/pipeline_baseline_hybrid_cage2
```
