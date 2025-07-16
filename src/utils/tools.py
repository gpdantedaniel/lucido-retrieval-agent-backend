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

def build_cheatsheet_story(title, concepts, definitions, equations, examples, references):
    """
    Generates a ReportLab-styles story
    returned as a list of Reportlab elements
    """
    story = []  # Create story content
    story.append(Paragraph(title, styles['CheatsheetTitle']))

    # Add content for each concept
    for concept in concepts:
        # Concept header
        story.append(Paragraph(concept, styles['Concept']))
        
        # Definition
        if concept in definitions:
            story.append(Paragraph("<b>Definition</b>", styles['Subheader']))
            story.append(Paragraph(definitions[concept], styles['Body']))
        
        # Equations
        if concept in equations and equations[concept]:
            story.append(Paragraph("<b>Equations</b>", styles['Subheader']))
            for eq in equations[concept]:
                story.append(Paragraph(eq, styles['Bullet']))
        
        # Examples
        if concept in examples and examples[concept]:
            story.append(Paragraph("<b>Examples</b>", styles['Subheader']))
            for ex in examples[concept]:
                story.append(Paragraph(ex, styles['Bullet']))

    # Add references like any good scientist
    story.append(Paragraph('<b>References</b>', styles['Concept']))
    for ref in references:
        story.append(Paragraph(ref, styles['Bullet']))
            
    return story

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

# ========== ReportLab Utilities ==========

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name='CheatsheetTitle',
    parent=styles['Heading1'],
    fontSize=20,
))

styles.add(ParagraphStyle(
    name='Concept',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.darkblue,
    spaceAfter=0.1*inch,
    spaceBefore=0.3*inch
))

styles.add(ParagraphStyle(
    name='Subheader',
    fontSize=10,
    textColor=colors.darkred,
    spaceAfter=0.1*inch
))

styles.add(ParagraphStyle(
    name='Body',
    parent=styles['BodyText'],
    fontSize=10,
    spaceAfter=0.1*inch
))

# Column dimensions
margin = 0.5 * inch
gutter = 0.2 * inch
col_width = (letter[0] - margin * 2 - gutter) / 2
col_height = letter[1] - margin * 2

two_column_template = PageTemplate(
    id='TwoColumns', 
    frames=[
        Frame(
            x1=margin,
            y1=margin,
            width=col_width,
            height=col_height,
            id='left'
        ),
        Frame(
            x1= margin + col_width + gutter,
            y1=margin,
            width=col_width,
            height=col_height,
            id='right'
        )
    ]
)

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

    # Create a buffer to receive doc
    buffer = BytesIO()

    doc = BaseDocTemplate(
        buffer, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin
    )
    
    # Give the document two columns
    doc.addPageTemplates([two_column_template])

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

    buffer = BytesIO()

    doc = BaseDocTemplate(
        buffer, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin
    )

    # Give the document two columns
    doc.addPageTemplates([two_column_template])

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

def build_quiz_story(title, questions, mapped_distractors, true_answers, references):
    """
    Generates a ReportLab story
    return as a list of ReportLab elements
    """
    # Store answers at the back 
    back_answers = []
    story = []
    
    story.append(Paragraph(title, styles['CheatsheetTitle']))

    if not (len(questions) == len(mapped_distractors) and len(mapped_distractors) == len(true_answers)):
        raise Exception('questions, distractors and true answers must all be of the same length')

    story.append(Paragraph('<b>Questions</b>', styles['Concept']))

    for question, distractors, true_answer in zip(questions, mapped_distractors, true_answers):
        shuffled = distractors + [true_answer]  # Not shuffled yet
        random.shuffle(shuffled)                # Now it is shuffled !
        shuffled = [f'{index + 1}. {answer}' for index, answer in enumerate(shuffled)]
        back_answers.append([a for a in shuffled if true_answer in a][0])

        story.append(Paragraph(question, styles['Body']))
        
        for ans in shuffled:
            story.append(Paragraph(ans, styles['Bullet']))

    # Send answers to the back (no peeking!)
    story.append(PageBreak())

    story.append(Paragraph('<b>Answers</b>', styles['Concept']))

    for question, ans in zip(questions, back_answers):
        story.append(Paragraph(question, styles['Body']))
        story.append(Paragraph(ans, styles['Bullet']))

    # Add references like any good scientist
    story.append(Paragraph('<b>References</b>', styles['Concept']))

    for ref in references:
        story.append(Paragraph(ref, styles['Bullet']))

    return story