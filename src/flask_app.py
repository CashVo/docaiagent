from flask import Flask, request, render_template
from llama_agent import build_rag_agent  # Import your RAG agent logic
from utils import init_bot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render a simple HTML UI

@app.route('/chat', methods=['POST'])
def chat():
    print("Start chatting...")
    user_input = request.json['user_input']
    response = build_rag_agent(user_input)
    html_response = f"<p> {response.response.replace("\n", "<br/>")} </p>"

    return { "response": html_response }


if __name__ == '__main__':
    init_bot()
    app.run(debug=True)
