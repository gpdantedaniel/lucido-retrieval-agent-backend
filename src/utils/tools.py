from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Frame, PageTemplate, BaseDocTemplate, PageBreak

from typing import List, Dict
from pydantic import BaseModel, Field
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
import tempfile
import datetime
from io import BytesIO
from uuid import uuid4
import random

from src.utils.state import State
from src.utils.utils.story_building import generate_doc, build_cheatsheet_story, build_quiz_story

# ========== Retrieval ==========

dense_retriever = Chroma(
    collection_name='bien200', 
    embedding_function=OllamaEmbeddings(model='dengcao/Qwen3-Embedding-0.6B:Q8_0'), 
    persist_directory='../../embeddings/chroma_db'
).as_retriever()

sparse_retriever = BM25Retriever.from_documents(CSVLoader(
    file_path='./data/processed/bien200_chunks.csv', 
    content_columns=['content'], 
    metadata_columns=['url', 'title', 'authors'],
    encoding='utf-8'
).load())

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# ========== Prompts ==========

COURSE_PROMPT = "You are a virtual teaching assistant and your job is to help students by answering relevant, theoretical questions."
SUMMARIZE_PROMPT = """Please summarize the following: {context}\n\nSummary:"""
SOURCE_FORMAT = '{authors}\n"{title}"\n{url}'

# ========== Functional Utilities ==========

summarize_chain = create_stuff_documents_chain(
    llm=ChatOllama(model='qwen3:14b', reasoning=False, temperature=0.6, top_p=0.95, top_k=20), 
    prompt=ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
)

def get_sources(docs) -> str:
    """
    Returns a string with all the sources
    formatted by title, authors and url.
    """
    sources = set() # Keep unique sources

    for doc in docs:
        metadata = doc.metadata
        source = SOURCE_FORMAT.format(
            title=metadata['title'],
            authors=metadata['authors'],
            url=metadata['url']
        )
        sources.add(source)

    # Concatenate into a string
    sources = '\n\n'.join(sources)
    return sources

container_name = 'outputs'
sas_expiry_time = 30 # in minutes

def upload_output(buffer) -> str:
    """
    Uploads a file in buffer as a PDF to Azure
    Blob Storage and returns a SAS url for reading.
    """
    # Generate a name for the blob
    blob_name=str(uuid4()) + '.pdf'

    # Establish connection to storage accoutn
    blob_service_client = BlobServiceClient.from_connection_string(os.environ['AZURE_BLOB_CONNECTION_STRING'])
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    blob_client.upload_blob(buffer, overwrite=True)

    # The URL can be accessed for 30 minutes
    start_time = datetime.datetime.now(datetime.timezone.utc)
    expiry_time = start_time + datetime.timedelta(minutes=sas_expiry_time)

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        account_key=os.environ['AZURE_STORAGE_ACCOUNT_KEY'],
        permission=BlobSasPermissions(read=True),
        start=start_time,
        expiry=expiry_time
    )

    sas_url = f'{blob_client.url}?{sas_token}'
    return sas_url

# ========== Input Schemas ==========

class CheatsheetInputSchema(BaseModel):
    title: str = Field(description='The main title of the cheatsheet')
    concepts: List[str] = Field(description='An ordered list of all concepts included in the cheatsheet. This helps in organizing the output.')
    definitions: Dict[str, str] = Field(description='A mapping from each concept to its definition where applicable')
    equations: Dict[str, List[str]] = Field(description='A mapping from each concept to a list of related equations where applicable.')
    examples: Dict[str, List[str]] = Field(description='A mapping from each concept to a list of illustrative examples where applicable.')
    state: Annotated[State, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

class QuizInputSchema(BaseModel):
    title: str = Field(description='The main title of the quiz')
    questions: List[str] = Field(description='An ordered list of all the questions.')
    distractor_answers: List[List[str]] = Field(description='An ordered list of lists of distractors answers for each question.')
    true_answers: List[str] = Field(description='An ordered list of the true answers for each question.')
    state: Annotated[State, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

# ========== Tools ==========

@tool('retrieval_tool', parse_docstring=True)
def retrieval_tool(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """
    Retrieve information from the knowledge base.
    Updates the 'retrievals' counter in the state.

    Args:
        query: the search query for retrieval-augmented generation
    """
    docs = ensemble_retriever.invoke(query)
    summary = summarize_chain.invoke({'context': docs, 'query': query})
    sources = get_sources(docs)

    return Command(update={
        'messages': [
            ToolMessage(
                content=summary, 
                artifact={'sources': sources}, 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool('cheatsheet_tool', args_schema=CheatsheetInputSchema)
def cheatsheet_tool(
    title: str, 
    concepts: List[str], 
    definitions: Dict[str, str], 
    equations: Dict[str, List[str]], 
    examples: Dict[str, List[str]],
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Generates a cheatsheet.
    Returns a success or failure message.
    When using, please respect the accepted argument types.
    Keep concept names consistent across arguments.
    And most importantly, be verbose!
    """
    # Generate document
    buffer = BytesIO()
    doc = generate_doc(buffer)

    # Extract potentially used references
    retrieval_calls = filter(lambda m: isinstance(m, ToolMessage) and m.name == 'retrieval_tool', state['messages'])
    references = list(set([source for call in retrieval_calls for source in call.artifact['sources'].split('\n\n')]))

    # Build the PDF document
    story = build_cheatsheet_story(title, concepts, definitions, equations, examples, references)
    doc.build(story)

    buffer.seek(0) # Reset once finished building
    download_url = upload_output(buffer) # Upload and get SAS download URL

    return Command(update={
        'messages': [
            ToolMessage(
                content='Cheatsheet successfully generated. Download link has been shared with user. Do not generate any download link.', 
                artifact={'download_url': download_url}, 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool('quiz_tool', args_schema=QuizInputSchema)
def quiz_tool(
    title: str,
    questions: List[str],
    distractor_answers: List[List[str]],
    true_answers: List[List[str]],
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Generates a multiple-choice quiz.
    Returns a success or failure message.
    When using, please respect the accepted argument types.
    Keep question ordering consistent across generated arguments.
    """
    # Generate document
    buffer = BytesIO()
    doc = generate_doc(buffer)

    # Extract potentially used references
    retrieval_calls = filter(lambda m: isinstance(m, ToolMessage) and m.name == 'retrieval_tool', state['messages'])
    references = list(set([source for call in retrieval_calls for source in call.artifact['sources'].split('\n\n')]))

    story = build_quiz_story(title, questions, distractor_answers, true_answers, references)
    doc.build(story)

    buffer.seek(0) # Reset once finished building
    download_url = upload_output(buffer) # Upload and get SAS download URL
    
    return Command(update={
        'messages': [
            ToolMessage(
                content='Quiz successfully generated. Download link has been shared with user. Do not generate any download link.', 
                artifact={'download_url': download_url}, 
                tool_call_id=tool_call_id
            )
        ]
    })