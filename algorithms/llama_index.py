from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    LangchainEmbedding,
    ServiceContext,
)
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from llama_index.prompts.prompts import SimpleInputPrompt
import torch

query_wrapper_prompt = SimpleInputPrompt(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)


class LlamaIndex:
    def __init__(self, domain_docs_dir=None, device="cuda", model_name="falcon", retrieval_enabled=True):  # noqa
        if retrieval_enabled and domain_docs_dir is None:
            raise RuntimeError(
                "'domain_docs_dir' argument must not be empty if "
                "'retrieval_enabled' is True")
        self.domain_docs_dir = domain_docs_dir
        self.device = device
        self.model_name = model_name
        self.retrieval_enabled = retrieval_enabled

    def load_model(self):
        if self.model_name == 'falcon':
            self.model_name = 'tiiuae/falcon-7b-instruct'

        if self.device == 'cuda':
            model_kwargs = {"torch_dtype": torch.float16,
                            "trust_remote_code": True}
        else:
            model_kwargs = {}

        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

        self.hf_predictor = HuggingFaceLLMPredictor(
            max_input_size=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
            model_kwargs=model_kwargs,
            tokenizer_outputs_to_remove=["token_type_ids"])

        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            chunk_size=512,
            llm_predictor=self.hf_predictor)

        if self.retrieval_enabled:
            documents = SimpleDirectoryReader(self.domain_docs_dir).load_data()
            new_index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context)

            # query with embed_model specified
            self.query_engine = new_index.as_query_engine(streaming=True)
        else:
            print("Retrieval disabled")
            self.query_engine = self.hf_predictor

        self.model_loaded = True

    def run_inference(self, prompt):
        if self.retrieval_enabled:
            return self.query_engine.query(prompt)
        else:
            return self.hf_predictor.predict(prompt)[0]
