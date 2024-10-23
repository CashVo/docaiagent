from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import get_bot_data

 # Initialize the RAG agent
def build_rag_agent (user_input):
    prompt = f"You are an AI assistant specializing in Pytorch. Answer this question: {user_input}"

    response = get_bot_data(name="query_engine").query(prompt)
    return response

if __name__ == "__main__":
    build_rag_agent()

