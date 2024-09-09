# flask_app.py

from flask import Flask, request, jsonify
from Question_answering_using_embeddings_Shopify import ShopifyAssistant

app = Flask(__name__)
assistant = ShopifyAssistant()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True)
    question = data['question']
    answer = assistant.ask(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 