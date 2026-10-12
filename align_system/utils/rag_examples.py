import json
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle, TextNode


class LlamaIndexExampleRetriever:
    def __init__(self, dataset_files, example_template, embedding_model,
                 query_template=None, retriever_kwargs=None, postprocessors=()):
        self.query_template = query_template
        self.postprocessors = postprocessors
        self._records = {}
        nodes = []
        for path in dict.fromkeys(Path(p).resolve() for p in dataset_files):
            with path.open() as stream:
                records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f'{path} must contain a JSON array of examples')
            for i, record in enumerate(records):
                record_id = f'{path}:{i}'
                self._records[record_id] = record
                # Keep labels and responses out of the searchable text.
                nodes.append(TextNode(id_=record_id, text=example_template(record)))
        if not nodes:
            raise ValueError('No examples were provided for retrieval')
        index = VectorStoreIndex(nodes, embed_model=embedding_model)
        options = {'similarity_top_k': 5, **(retriever_kwargs or {})}
        self._retriever = index.as_retriever(**options)

    def __call__(self, query):
        if not isinstance(query, str) and self.query_template is not None:
            query = self.query_template(query)
        if not isinstance(query, str):
            raise TypeError('A structured query requires a query_template returning text')
        bundle = QueryBundle(query_str=query)
        nodes = self._retriever.retrieve(bundle)
        for processor in self.postprocessors:
            nodes = processor.postprocess_nodes(nodes, query_bundle=bundle)
        return [{'id': node.node.node_id, 'record': self._records[node.node.node_id],
                 'score': node.score} for node in nodes]
