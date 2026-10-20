"""A lazy BERTScore model shared by one ICL generator factory's attributes."""


class BatchedBERTScorer:
    """Keep one model on the GPU and explicitly bound each scoring batch.

    Instantiate this once as ``icl_generator_partial.scorer``. Hydra's partial
    then passes the same object to every attribute generator. Loading is lazy
    so composing configs and creating unused generators do not allocate a GPU.
    BERTScore's score() batch size must be passed explicitly: its constructor's
    batch_size does not control scoring in the installed BERTScore version.
    """

    def __init__(self, batch_size=32, lang='en', device=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('BERT batch_size must be a positive integer')
        self.batch_size = batch_size
        self.lang = lang
        self.device = device
        self._scorer = None

    def score(self, cands, refs):
        if self._scorer is None:
            from bert_score import BERTScorer
            self._scorer = BERTScorer(lang=self.lang, device=self.device)
        return self._scorer.score(cands, refs, batch_size=self.batch_size)
