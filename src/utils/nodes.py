from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage

from src.utils.state import State
from src.utils.tools import retrieval_tool, cheatsheet_tool

# Configuration by https://huggingface.co/Qwen/Qwen3-8B

model = ChatOllama(model='qwen3:8b', reasoning=True, temperature=0.6, top_p=0.95, top_k=20)
tools = [retrieval_tool, cheatsheet_tool]
max_retrievals = 3

# ========== Prompts and messages ==========

COURSE_PROMPT = "You are Lucido, a virtual teaching assistant."

HOME = f"""{COURSE_PROMPT}
Do not answer questions using your own knowledge.
Use {retrieval_tool.name} to retrieve information from your knowledge base.
Use {cheatsheet_tool.name} to generate a cheatsheet if requested."""

# ========== Nodes ==========

def init_defaults(state: State):
    """
    Initializes default values. This is a temporary
    fix that eases workflows with LangGraph Studio
    """
    return {"retrievals": 0}

def home(state: State):
    """
    Determines whether retrieval is needed.
    If the user's query can be answered, an answer is generated.
    Otherwise, the retrieval tool is called.
    """
    
    sys_message = SystemMessage(content=HOME)
    response = model.bind_tools(tools).invoke([sys_message] + state['messages'])

    last_message = state['messages'][-1]

    if len(response.tool_calls) == 0:
        # If a knowledge retrieval tool call was made
        if isinstance(last_message, ToolMessage) and last_message.name == 'retrieval_tool':
            sources = last_message.artifact['sources']
            response.content = f'{response.content}\n\nSources:\n\n{sources}'

    return {'messages': [response]}