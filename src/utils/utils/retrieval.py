import os
import json
import urllib.request
from typing import List
import pandas as pd
from io import StringIO
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from azure.storage.blob import BlobServiceClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import chromadb

# ========== Utilities ==========

class AzureMLEndpointEmbeddings(Embeddings):
    """Azure ML Endpoint integration for generating embeddings"""
    def __init__(self):
        pass
    
    def get_embeddings(self, text: str) -> List[float]:
        """Gets embeddings from Azure ML Endpoint"""
        body = str.encode(json.dumps({'text': text}))
        headers = {'Content-Type':'application/json', 'Accept': 'application/json', 'Authorization':('Bearer '+ os.getenv('AZURE_ML_KEY'))}
        
        req = urllib.request.Request(os.getenv('AZURE_ML_EMBEDDING_ENDPOINT'), body, headers)
        response = urllib.request.urlopen(req)
        result = response.read()

        embeddings = json.loads(json.loads(result))['embeddings']
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds search documents"""
        return [self.get_embeddings(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embeds search queries"""
        return self.get_embeddings(text)
    
embeddings_client = AzureMLEndpointEmbeddings()

def load_docs_from_blob():
    """
    Loads the CSV chunks from Azure Blob Storage
    and returns them as list of LangChain documents.
    """
    # Azure connection details
    connection_string = os.getenv('AZURE_BLOB_CONNECTION_STRING')
    container_name = "knowledge"
    blob_name = "bien200_chunks.csv"

    # Create BlobServiceClient and download blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download blob content as text (decoded from bytes)
    downloaded_blob = blob_client.download_blob()
    csv_content = downloaded_blob.content_as_text()

    # Convert CSV string to DataFrame
    df = pd.read_csv(StringIO(csv_content))

    docs = []

    for _, row in df.iterrows():
        docs.append(Document(
        page_content=row['content'],
        metadata={
            'url': row['url'], 
            'title': row['title'], 
            'authors': row['authors']
        }
    ))
        
    return docs

# ========== Retrievers ==========

chroma_client = chromadb.CloudClient(
    tenant=os.getenv('CHROMA_TENANT'),
    database=os.getenv('CHROMA_DATABASE'),
    api_key=os.getenv('CHROMA_API_KEY')
)

dense_retriever = Chroma(
    collection_name='bien200', 
    embedding_function=embeddings_client, 
    client=chroma_client
).as_retriever(search_kwargs={'k': 4})

sparse_retriever = BM25Retriever.from_documents(documents=load_docs_from_blob(), k=4)

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)