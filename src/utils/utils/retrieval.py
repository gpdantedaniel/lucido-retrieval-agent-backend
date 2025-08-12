import os
import csv
from io import StringIO
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from azure.storage.blob import BlobServiceClient
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
import chromadb

# ========== Utilities ==========
    
embeddings_client = AzureOpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

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

    docs = []

    # Parse CSV string with csv.DictReader
    reader = csv.DictReader(StringIO(csv_content))

    for row in reader:
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