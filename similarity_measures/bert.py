import BERTSimilarity.BERTSimilarity as bertsimilarity


# Needed to silence BERT warning messages, see: https://stackoverflow.com/questions/67546911/python-bert-error-some-weights-of-the-model-checkpoint-at-were-not-used-when # noqa
from transformers import logging
logging.set_verbosity_error()


def build_bert_similarity_measure_func():
    bertsim = bertsimilarity.BERTSimilarity()

    return bertsim.calculate_distance
