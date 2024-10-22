from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model import load_llama_model
from ingestion import ingest_web_content
from content_sources import content_urls

 # Initialize the RAG agent
def build_rag_agent (user_input):
    # Specify what model to use for this agent
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
  
    model_data = load_llama_model(model_name)
    documents = ingest_web_content(content_urls())

    Settings.llm = model_data["model"]
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    prompt = f"You are an AI assistent specalizing in Pytorch. Answer this question: {user_input}"

    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)
    response = query_engine.query(prompt)
    return response

if __name__ == "__main__":
    build_rag_agent()

