from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from flask import Flask, request, jsonify, render_template

retriever = Chroma(
    collection_name='main', 
    embedding_function=OllamaEmbeddings(model='dengcao/Qwen3-Embedding-0.6B:F16'), 
    persist_directory='../../embeddings/vrmuseum'
).as_retriever(search_kwargs={'k': 5})

app = Flask(__name__)

@app.route('/semantic_search', methods=['GET'])
def semantic_search():
    """
    Semantic search function that runs on 
    Qwen Embeddings and a local ChromaDB vector store
    """
    query = request.args.get('query')

    if not query:
        return jsonify({'error': 'No query parameter provided'}), 400
    
    try:
        results = retriever.invoke(query)
        serializable_results = [dict(doc) for doc in results]
        return jsonify({'query': query, 'results': serializable_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)