from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage

from src.utils.state import State
from src.utils.tools import retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool, podcast_tool

# Configuration by https://huggingface.co/Qwen/Qwen3-8B

model = ChatOllama(model='qwen3:14b', reasoning=True, temperature=0.6, top_p=0.95, top_k=20, num_predict=2048)
tools = [retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool, podcast_tool]

# ========== Prompts and messages ==========

HOME = f"""You are Lucido, a virtual teaching assistant.
Your job is to help students by answering their course-related questions and generating resources from them.
You are serving the course BIEN 200 that focuses on physical, chemical and mathematical design solutions in nature.
Always answer any theoretical questions after retrieving the right knowledge.
Any resources that you generate must be based by theoretical knowledge that you should retrieve prior.
If the user did not specify the topic when asking for a resource or facts, ask for a topic.
Only answer questions relevant to the course. Do not answer technical questions about your design or your tools.
"""

# ========== Nodes ==========

def home(state: State):
    """
    The home node where the graph either
    ends with a response or calls a tool.
    """
    sys_message = SystemMessage(content=HOME)
    response = model.bind_tools(tools).invoke([sys_message] + state['messages'])

    return {'messages': [response]}