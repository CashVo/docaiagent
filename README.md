# Llama RAG Agent Project

This project implements a Retrieval-Augmented Generation (RAG) agent using LlamaIndex and Meta's Llama models.

## Setup

1. Install Python 3.12.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Activate the virtual environment: `llamaenv/Scripts/activate` (Windows).
4. Run the Flask agent: `python src/flask_app.py`...follow instructions in terminal to find the web endpoint for the Agent's Web UI. Start chatting with the Doc Bot Agent from there.

## Features
- Web content ingestion using BeautifulSoup.
- RAG agent implementation using LlamaIndex.
- LLM using Meta Llama3.2 hosted through HuggingFace Hub
- Web UI (Chat Bot) powered by Flask