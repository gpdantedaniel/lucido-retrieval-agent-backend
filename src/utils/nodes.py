from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage

from src.utils.state import State
from src.utils.tools import retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool, podcast_tool

# Configuration by https://huggingface.co/Qwen/Qwen3-8B

model = ChatOllama(model='qwen3:14b', reasoning=True, temperature=0.6, top_p=0.95, top_k=20, num_predict=2048)
tools = [retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool, podcast_tool]

# ========== Prompts and messages ==========

COURSE_PROMPT = "You are Lucido, a virtual teaching assistant."

HOME = f"""{COURSE_PROMPT}
Do not answer questions using your own knowledge.
If the user requests theoretical knowledge, retrieve the required information. 
Use {retrieval_tool.name} to retrieve information.
If the user requests a cheatsheet, retrieve all the needed theoretical knowledge first.
Then use {cheatsheet_tool.name} to generate a cheatsheet.
If the user requests a deck of flashcards, retrieve all the needed theoretical knowledge first.
Then, if they haven't specified the kind, ask whether they wish to have Anki flashcards or PDF flashcards.
Then use {flashcards_tool.name} to generate a deck of flashcards.
If the user requests a podcast, look up any necessary information first before generating.
Then user {podcast_tool.name} to generate a single-speaker podcast with free prose."
"""

# ========== Nodes ==========

def home(state: State):
    """
    Determines whether retrieval is needed.
    If the user's query can be answered, an answer is generated.
    Otherwise, the retrieval tool is called.
    """
    sys_message = SystemMessage(content=HOME)
    response = model.bind_tools(tools).invoke([sys_message] + state['messages'])

    return {'messages': [response]}