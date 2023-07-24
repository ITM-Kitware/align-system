from algorithms.llama_index import LlamaIndex

DOCUMENTS_DIR = '/data/david.joy/ITM/data/DomainDocumentsPDF'

llama_index = LlamaIndex(domain_docs_dir=DOCUMENTS_DIR)
llama_index.load_model()

def retrieve_texts(llama_index, query):
    query_engine = llama_index.index.as_query_engine(response_mode='no_text')
    response = query_engine.query(query)

    retrieved_texts = [node.node.text for node in response.source_nodes]
    return retrieved_texts

prompts = [
    "What is the first step in applying a tourniquet?",
    "How do you perform the Heimlich maneuver on an adult?",
    "What should you do if someone is having a seizure?",
    "What are the signs of internal bleeding?",
    "What are the techniques for wound cleansing?",
    "How can you tell if someone has a concussion?",
    "When should a splint be used?",
    "What should you do if someone has ingested poison?",
    "What are the indications for using a chest seal?",
    "What is the correct procedure for CPR?",
    "How should you treat a burn wound in the field?",
    "What should be done if someone is in shock?",
    "How to handle a broken bone on the battlefield?",
    "What are the steps for treating a bullet wound?",
    "What are the guidelines for using morphine in battlefield casualties?",
    "What is the protocol for treating heat stroke?",
    "How do you identify and treat hypothermia on the field?",
    "What are the methods of stopping bleeding in the field?",
    "What are the signs of a traumatic brain injury?",
    "What should you do if someone is experiencing difficulties in breathing?"
]

with open("outputs/retrieval_inspector_report.md", "w") as outfile:
    for prompt in prompts:
        outfile.write(f'## Prompt\n<pre>{prompt}</pre>\n\n')

        # Retrieval
        retrieved_texts = retrieve_texts(llama_index, prompt)
        outfile.write('## Retrieved Texts\n')
        for text in retrieved_texts:
            outfile.write(f'<pre>{text}</pre>\n')
        outfile.write('\n')

        # Inference
        outfile.write('## Model Response\n')
        model_response = llama_index.run_inference(prompt)
        outfile.write(f'<pre>{model_response}</pre>\n\n')

print('DONE')