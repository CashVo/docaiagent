import requests
from bs4 import BeautifulSoup as bs
from llama_index.core import Document

# Handle web content scraping and data ingestion
def ingest_web_content(urls):
    documents = []
    for url in urls:
        response = requests.get(url)
        soup = bs(response.content, "html.parser")
        text = soup.get_text()
        documents.append(Document(text=text))
    return documents
