"""Verify GPU model sharing and batching through the real Hydra factory."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from align_system.utils.bert_scorer import BatchedBERTScorer
from align_system.utils.incontext_utils import bert_similarity_selection


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError, match='positive integer'):
        BatchedBERTScorer(batch_size=size)


def test_hydra_partial_shares_lazy_model_and_passes_batch_thirty_two(monkeypatch):
    from align_system.utils.incontext_utils import Phase2ComparativeRegressionIncontextExampleGenerator

    constructed, batches = [], []

    class Backend:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def score(self, cands, refs, batch_size):
            batches.append(batch_size)
            scores = torch.arange(len(cands), dtype=torch.float32)
            return scores, scores, scores

    monkeypatch.setattr('bert_score.BERTScorer', Backend)
    monkeypatch.setattr(Phase2ComparativeRegressionIncontextExampleGenerator,
                        'set_icl_datasets', lambda self: None)
    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_comparative_regression_live'])
    factory = instantiate(cfg.driver.extra_tools.comparative_regression.icl.icl_generator_partial)
    generators = [factory(target_kdmas=[{'kdma': name}])
                  for name in ('medical', 'merit', 'affiliation', 'search', 'personal_safety')]
    assert len({id(g.scorer) for g in generators}) == 1
    assert not constructed
    candidates = [{'index': n} for n in range(25)]
    for generator in generators:
        result = bert_similarity_selection(candidates, ['example'] * 25, 'query',
                                           20, scorer=generator.scorer)
        assert [r['index'] for r in result] == list(range(24, 4, -1))
    assert constructed == [{'lang': 'en', 'device': 'cuda'}]
    assert batches == [32] * 5
