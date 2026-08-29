'''
Uses the LangChain framework to enable prompt-injected RAG with hydra-configurable
inference LLM model selection. The knowledge base vector store is dynamically
generated from the provided document files.

Prompt-Injection RAG ADM Components
'''
from typing import Iterable, Union, List, Dict
from os import PathLike

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from align_system.algorithms.abstracts import ADMComponent

DocumentFileType = Union[str, bytes, PathLike]
DocumentFileListType = Iterable[DocumentFileType]


class LangChainRAGIndexerADMComponent(ADMComponent):
    def __init__(self,
                 docs_files: DocumentFileListType,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 add_start_index: bool = True,
                 k: int = 5):
        self.docs_files = docs_files
        self.embedding_model_name = embedding_model_name
        self.k = k

        docs = LangChainRAGIndexerADMComponent._load_docs(docs_files)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )
        all_splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.from_documents(all_splits, embeddings)

        self.retriever = vector_store.as_retriever(search_kwargs={"k": k})

    def run_returns(self):
        return "rag_context"

    def run(self, scenario_state) -> str:
        query = scenario_state.unstructured
        docs = self.retriever.invoke(query)
        passages = [f"[Passage {i}]\n{doc.page_content.strip()}"
                    for i, doc in enumerate(docs, start=1)]
        return "\n\n".join(passages)

    def retrieve(self, query: str) -> Dict:
        docs = self.retriever.invoke(query)
        return {
            "question": query,
            "context": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        }

    @staticmethod
    def _load_docs(docs_files: DocumentFileListType) -> List[Document]:
        docs = []
        for d in docs_files:
            with open(d, 'r') as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": str(d)}))
        return docs
