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
