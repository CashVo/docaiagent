import requests
from bs4 import BeautifulSoup as bs
from llama_index.core import Document

# Handle web content scraping and data ingestion
def ingest_web_content(urls):
    documents = []
    print("Loading data...")
    for url in urls:
        response = requests.get(url)
        soup = bs(response.content, "html.parser")
        p_text = soup.find_all(name=['p', 'h1', 'h2', 'h3', 'pre']) # 
        full_content = ""
        for text in p_text:
            full_content = full_content + '\n' + text.get_text()
        print(f"Text content from url {url}\n {full_content}")
        documents.append(Document(text=full_content))
    
    print("Data loaded.")
    return documents
