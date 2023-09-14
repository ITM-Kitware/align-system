from bert_score import score

# Silences warning messages
from transformers import logging
logging.set_verbosity_error()


def bert_score_similarity_f1(str1, str2):
    # See: https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb

    p, r, f1 = score([str1], [str2], lang='en')

    return f1


def bert_score_similarity_recall(str1, str2):
    # See: https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb

    p, r, f1 = score([str1], [str2], lang='en')

    return r


def bert_score_similarity_precision(str1, str2):
    # See: https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb

    p, r, f1 = score([str1], [str2], lang='en')

    return p
