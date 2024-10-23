# from llama_index.core import VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import get_bot_data

 # Initialize the RAG agent
def prompt_rag_agent (user_input):
    prompt = f"You are an AI assistant specializing in Pytorch. Answer this question: {user_input}"

    query_engine = get_bot_data(name="query_engine")
    response = query_engine.query(prompt)
    return response

if __name__ == "__main__":
    prompt_rag_agent()

