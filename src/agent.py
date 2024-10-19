from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model import load_llama_model
from ingestion import ingest_web_content
from content_sources import content_urls

 # Initialize the RAG agent
def build_rag_agent (user_input):
    # Specify what model to use for this agent
    model_name = "meta-llama/Llama-3.2-1B"
  
    model_data = load_llama_model(model_name)
    documents = ingest_web_content(content_urls())

    Settings.llm = model_data["model"]
    print("Start embedding...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    print("Start indexing...")
    index = VectorStoreIndex.from_documents(documents=documents)
    print("Start query engine...")
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)
    print(f"Querying for: {user_input}")
    response = query_engine.query(user_input)
    print(f"got response: {response}")
    return response

if __name__ == "__main__":
    build_rag_agent()

