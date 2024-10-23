from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings. huggingface import HuggingFaceEmbedding
from llama_index.core import  Settings

# Load the Llama model
def load_llama_model():
    model_data = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "model_embedding": "BAAI/bge-m3"
    }
    tokenizer = AutoTokenizer.from_pretrained(model_data["model_name"])
    llm = HuggingFaceLLM(
            model_name=model_data["model_name"],
            tokenizer=tokenizer
        )

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_data["model_embedding"])

