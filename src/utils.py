from llama_index.core import VectorStoreIndex, Settings
from model import load_llama_model
from ingestion import ingest_web_content
from content_sources import content_urls
 
bot_data = {} # Stores a list of objects about this bot

def get_bot_data(name):
    return bot_data[name]

def init_bot():
    """Initialize the bot with the following tasks:
    1) Load content
    2) Index content with LlamaIndex
    3) Set embedding

    Finally, stores the initialized objects into the bot_data list
    """

    print("Initializing...")
    load_llama_model()
    documents = ingest_web_content(content_urls())
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)
    
    # Save context obj
    bot_data["Settings"] = Settings
    bot_data["documents"] = documents
    bot_data["index"] = index
    bot_data["query_engine"] = query_engine
    print("Bot initialized...")
