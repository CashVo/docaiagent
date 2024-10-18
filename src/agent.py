from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model import load_llama_model
from ingestion import ingest_web_content

 # Initialize the RAG agent
def build_rag_agent (urls):
    # Specify what model to use for this agent
    model_name = "meta-llama/Llama-3.2-1B"

    model_data = load_llama_model(model_name)
    documents = ingest_web_content(urls)

    Settings.llm = model_data["model"]
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)

    while True:
        query = input("What's your question? ")
        response = query_engine.query(query)
        print(f"Agent response: {response.response}")

if __name__ == "__main__":
    urls = [
        "https://pytorch.org/tutorials/beginner/basics/intro.html",
        "https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html",
        "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
        "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"
    ]
    build_rag_agent(urls)

