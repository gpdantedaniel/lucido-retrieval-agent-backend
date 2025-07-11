from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import HRFlowable
from uuid import uuid4
from pathlib import Path
import tempfile

from src.utils.state import State

# Chunk parameters
chunk_size = 256
chunk_overlap = 32

"""
embedding_model = OllamaEmbeddings(model='dengcao/Qwen3-Embedding-8B:Q8_0')
embeddings_path = f'./outputs/hyperbook_embeddings_{chunk_size}_{chunk_overlap}'

# Loading the vector store locally
vectorstore = InMemoryVectorStore.load(path=embeddings_path, embedding=embedding_model)
dense_retriever = vectorstore.as_retriever()
"""

docs = CSVLoader(
    file_path='./data/raw/hyperbook.csv', 
    content_columns=['content'], 
    metadata_columns=['url', 'title', 'authors'],
    encoding='utf-8'
).load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(docs)

sparse_retriever = BM25Retriever.from_documents(docs)

"""
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)
"""

ensemble_retriever = sparse_retriever   # A temporary fix for quick prototyping

COURSE_PROMPT = "You are a virtual teaching assistant and your job is to help students by answering relevant, theoretical questions."
summarizer_model = ChatOllama(model='qwen3:8b', reasoning=False, temperature=0)

SUMMARIZE_PROMPT = """Please summarize the following: {context}\n\nSummary:"""
SOURCE_FORMAT = '{title}\n{authors}\n{url}'

summarize_prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
summarize_chain = create_stuff_documents_chain(summarizer_model, summarize_prompt)

SOURCE_FORMAT = '{title}\n{authors}\n{url}'

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

@tool('retrieval_tool', parse_docstring=True)
def retrieval_tool(query: str, tool_call_id: Annotated[str, InjectedToolCallId], state: Annotated[State, InjectedState]) -> str:
    """
    Retrieve information from the knowledge base.
    Updates the 'retrievals' counter in the state.

    Args:
        query: the search query for retrieval-augmented generation
    """
    docs = ensemble_retriever.invoke(query)
    summary = summarize_chain.invoke({'context': docs, 'query': query})
    sources = get_sources(docs)

    retrievals = state.get('retrievals', 0)

    return Command(update={
        'retrievals': retrievals + 1,
        'messages': [
            ToolMessage(
                content=summary, 
                artifact={'sources': sources}, 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool('cheatsheet_tool', parse_docstring=True)
def cheatsheet_tool(title: str, body: str) -> str:
    """
    Generates a cheatsheet locally.
    Returns a URL that that provides access for downloading.

    Args:
        title: the title of the cheatsheet
        body: the body of the cheatsheet's content
    """

    # Create a temporary directory
    filename = './cheatsheet.pdf'

    doc = SimpleDocTemplate(filename, pagesize=LETTER,
                            leftMargin=72, rightMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=0)
    body_style = styles['BodyText']
    
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color="black"))
    story.append(Spacer(1, 12))
    
    for para in body.split("\n\n"):
        story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 12))

    doc.build(story)

    # Return the file URL
    return filename

