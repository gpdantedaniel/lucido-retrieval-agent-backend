import os
import json
from sentence_transformers import SentenceTransformer

def init():
    global model, tokenizer
    # The model and tokenizer are loaded from a path where the model is registered in Azure
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'Qwen3-Embedding-06B')
    model = SentenceTransformer(model_path)

def run(raw_data):
    try:
        # Extract text from dictionary
        text = json.loads(raw_data)['text']
        predictions = model.encode(text)
        predictions = predictions.tolist()

        # Return predictions in a JSON format
        return json.dumps({"embeddings": predictions})

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})