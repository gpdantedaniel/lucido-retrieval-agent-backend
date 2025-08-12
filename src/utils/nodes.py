import os
from langchain_core.messages import SystemMessage
from src.utils.state import HomeInputState
from src.utils.tools import retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool
from langchain_openai import AzureChatOpenAI

# The below can be uncommented when Summarization becomes necessary
# from langmem.short_term import SummarizationNode
# from langchain_core.messages.utils import count_tokens_approximately

chat_model = summarizer_model = AzureChatOpenAI(
    azure_deployment='gpt-5-nano', 
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY')
)

tools = [retrieval_tool, cheatsheet_tool, quiz_tool, flashcards_tool]

# ========== Prompts and messages ==========

sys_message = SystemMessage(content=f"""You are Lucido, a virtual teaching assistant.
You are serving the course BIEN 200 which focuses on physical, chemical and mathematical design solutions in nature.
Your job is to help students by answering their course-related questions and generating educational resources for them.
If the user did not specify the topic when asking for a resource or facts, ask for a topic.
When answering or generating resources, **always** and **only** use retrieved knowledge.
You should avoid using your own knowledge if it is not backed by retrieved knowledge.
When retrieving knowledge, search for any relevant, specific design solutions.
If necessary, try breaking up the query into multiple queries.
Your knowledge base consists of papers written by students about organisms and their design solutions.
When using tools, please respect and consider their accepted argument types as well as any instruction that comes with them.
When creating learning resources, be verbose when generating their content to make rich study aids. Otherwise, be concise.""")

# ========== Nodes ==========

def home(state: HomeInputState):
    """
    The home node where the graph either
    ends with a response or calls a tool.
    """
    response = chat_model.bind_tools(tools).invoke([sys_message] + state['messages'])
    
    return {'messages': [response]}

"""
summarization_node = SummarizationNode(
    model=summarizer_model,
    max_tokens=2048,
    max_tokens_before_summary=16384,
    max_summary_tokens=1024,
    token_counter=count_tokens_approximately,
    input_messages_key='messages',
)
"""