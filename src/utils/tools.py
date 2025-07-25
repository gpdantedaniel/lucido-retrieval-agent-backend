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
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from typing import List, Dict
from pydantic import BaseModel, Field
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
import datetime
from io import BytesIO
from uuid import uuid4
import random
import genanki
from kokoro import KPipeline
import numpy as np
import soundfile as sf
import ollama
from src.utils.state import State
from src.utils.utils.story_building import generate_doc, generate_flashcard_doc, build_cheatsheet_story, build_quiz_story, build_flashcards_story

# ========== Retrieval ==========

dense_retriever = Chroma(
    collection_name='bien200', 
    embedding_function=OllamaEmbeddings(model='dengcao/Qwen3-Embedding-0.6B:Q8_0'), 
    persist_directory='./embeddings/chroma_db'
).as_retriever(search_kwargs={'k': 4})

sparse_retriever = BM25Retriever.from_documents(
    documents=CSVLoader(
        file_path='./data/processed/bien200_chunks.csv', 
        content_columns=['content'], 
        metadata_columns=['url', 'title', 'authors'],
        encoding='utf-8',
    ).load(),
    k=4
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# ========== Prompts and Templates ==========

COURSE_PROMPT = "You are a virtual teaching assistant and your job is to help students by answering relevant, theoretical questions."
SUMMARIZE_PROMPT = """Please summarize the following: {context}\n\nSummary:"""
RETRIEVAL_FORMAT = 'title: {title}\ncontent: {content}'
SOURCE_FORMAT = '{authors}\n"{title}"\n{url}'

# ========== Functional Utilities ==========

def summarize_docs(docs):
    """
    Summarizes the context of a list
    of documents using an LLM call
    """
    context = '\n'.join(RETRIEVAL_FORMAT.format(title=doc.metadata['title'], content=doc.page_content) for doc in docs)
    prompt = SUMMARIZE_PROMPT.format(context=context)
    response = ollama.chat(
        model='qwen3:14b',
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        think=False
    )

    return response['message']['content']

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

def upload_output(buffer, file_extension) -> str:
    """
    Uploads a file in buffer to Azure
    Blob Storage and returns a SAS url for reading.
    """
    # Generate a name for the blob
    blob_name=str(uuid4()) + f'.{file_extension}'

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

def generate_model_deck(title):
    """
    Generates Anki model and deck
    and returns them with unique int IDs
    """
    assign_id = lambda: random.randint(1, 1_000_000_000)
    model = genanki.Model(
        model_id=assign_id(),
        name='Default',
        fields = [
            {'name': 'Question'},
            {'name': 'Answer'}
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}'
            }
        ]
    )

    deck = genanki.Deck(deck_id=assign_id(), name=title)
    return model, deck

# ========== Input Schemas ==========

class CheatsheetInputSchema(BaseModel):
    title: str = Field(description='The main title of the cheatsheet')
    concepts: List[str] = Field(description='An ordered list of all concepts included in the cheatsheet. This helps in organizing the output.')
    definitions: List[str] = Field(description='A mapping from each concept to its definition where applicable')
    equations: List[List[str]] = Field(description='A mapping from each concept to a list of related equations where applicable.')
    examples: List[List[str]] = Field(description='A mapping from each concept to a list of illustrative examples where applicable.')
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

class QuizInputSchema(BaseModel):
    title: str = Field(description='The main title of the quiz')
    questions: List[str] = Field(description='An ordered list of all the questions.')
    distractor_answers: List[List[str]] = Field(description='An ordered list of lists of distractors answers for each question.')
    true_answers: List[str] = Field(description='An ordered list of the true answers for each question.')
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

class FlashcardsInputSchema(BaseModel):
    title: str = Field(description='The title of the flashcards deck')
    questions: List[str] = Field(description='An ordered list of all the questions.')
    answers: List[str] = Field(description='An ordered list of all the answers aligned with the questions.')
    generate_anki: bool = Field(description='Whether an Anki deck should be generated.')
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

class PodcastInputSchema(BaseModel):
    content: str = Field(description='The free prose content of the podcast.')
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]
    
# ========== Tools ==========

@tool('retrieval_tool')
def retrieval_tool(
    query: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """
    Retrieves information from the knowledge base.
    Use if theoretical knowledge is required for a task.
    Please respect the accepted argument types.
    """
    docs = ensemble_retriever.invoke(query)
    summary = summarize_docs(docs)
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
    definitions: List[str], 
    equations: List[List[str]], 
    examples: List[List[str]],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Generates a cheatsheet.
    Use if the user requests one.
    Please respect the accepted argument types.
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
    download_url = upload_output(buffer, 'pdf') # Upload and get SAS download URL

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
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Generates a multiple-choice quiz.
    Use if the user requests one.
    Please respect the accepted argument types.
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
    download_url = upload_output(buffer, 'pdf') # Upload and get SAS download URL
    
    return Command(update={
        'messages': [
            ToolMessage(
                content='Quiz successfully generated. Download link has been shared with user. Do not generate any download link.', 
                artifact={'download_url': download_url}, 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool('flashcards_tool', args_schema=FlashcardsInputSchema)
def flashcards_tool(
    title: str,
    questions: List[str],
    answers: List[str],
    generate_anki: bool,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Generates PDF or Anki Flashcards.
    Use if the user requests some after specifying the type (PDF or Anki).
    Please respect the accepted argument types.
    """
    # Generate document
    buffer = BytesIO()

    # Extract potentially used references
    retrieval_calls = filter(lambda m: isinstance(m, ToolMessage) and m.name == 'retrieval_tool', state['messages'])
    references = list(set([source for call in retrieval_calls for source in call.artifact['sources'].split('\n\n')]))
    
    if generate_anki:
        model, deck = generate_model_deck(title)
        for q, a in zip(questions, answers):
            note = genanki.Note(model=model, fields=[q, a])
            deck.add_note(note)
        genanki.Package(deck).write_to_file(buffer)

        buffer.seek(0) # Reset once finished building
        download_url = upload_output(buffer, 'apkg')
    else:
        doc = generate_flashcard_doc(buffer, title)
        story = build_flashcards_story(questions, answers)
        doc.build(story)

        buffer.seek(0) # Reset once finished building
        download_url = upload_output(buffer, 'pdf')

    return Command(update={
        'messages': [
            ToolMessage(
                content='Flashcards successfully generated. Download link has been shared with user. Do not generate any download link.', 
                artifact={'download_url': download_url}, 
                tool_call_id=tool_call_id
            )
        ]
    })

@tool('podcast_tool', args_schema=PodcastInputSchema)
def podcast_tool(
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Generates a podcast.
    Use if the user requests one.
    Please respect the accepted argument types.
    """
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(content, voice='af_heart', split_pattern=None)
    audio = np.concatenate([segment[-1] for segment in generator])

    buffer = BytesIO()
    sf.write(buffer, audio, 24_000, format='WAV', subtype='PCM_24')
    buffer.seek(0)

    download_url = upload_output(buffer, 'wav')

    return Command(update={
        'messages': [
            ToolMessage(
                content='Podcast successfully generated. Download link has been shared with user. Do not generate any download link.', 
                artifact={'download_url': download_url}, 
                tool_call_id=tool_call_id
            )
        ]
    })