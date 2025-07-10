from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage

from src.utils.state import State
from src.utils.tools import retrieval_tool, cheatsheet_tool

max_retrievals = 3
model = AzureChatOpenAI(azure_deployment='gpt-4o', api_version='2024-10-21')

# ========== Prompts ==========

COURSE_PROMPT = "You are a virtual teaching assistant and your job is to help students by answering relevant, theoretical questions."

DECIDE_RETRIEVAL_PROMPT = f"""{COURSE_PROMPT}
If the student's question is not relevant, remind them that their questions should be course-related.
If you can answer the question of the student based information found in on the conversation, do so.
If the conversation does not have the information needed, look up answers using the {retrieval_tool.name}. Do not use your own training knowledge.
Consider what the user is asking as well as the context to formulate a query when using the {retrieval_tool.name}
If the user needs to have a cheatsheet generated baed on the content of the conversation, use {cheatsheet_tool.name}."""

DECIDE_RELEVANCE_PROMPT = f"""{COURSE_PROMPT}
If the results provided by the tool have the answer to the user's query, only generate an answer based on these results. Do not generate sources.
If not, refine your query based and use the tool {retrieval_tool.name} again."""

RETRIEVAL_FAILED_PROMPT = f"""{COURSE_PROMPT}
If the results provided by the tool have the answer to the user's query, generate an answer based on these results.
If not, state that you were unable to get the required information and ask the user to refine their question."""

# ========== Nodes ==========

def init_defaults(state: State):
    """
    Initializes default values. This is a temporary
    fix that eases workflows with LangGraph Studio
    """
    return {"retrievals": 0}

def retrieval_needed(state: State):
    """
    Determines whether retrieval is needed.
    If the user's query can be answered, an answer is generated.
    Otherwise, the retrieval tool is called.
    """
    
    sys_message = SystemMessage(content=DECIDE_RETRIEVAL_PROMPT)
    tools = [retrieval_tool]
    response = model.bind_tools(tools).invoke([sys_message] + state['messages'] )
    
    return {'messages': [response]}

def judge_results(state: State):
    """
    Determines whether the tool's results are relevant.
    If they are, moves on to generating an answer.
    If they are not, calls the tool again with a refined query.
    If retrieval exceeds 3 calls, returns a failure message.
    """
    retrievals = state.get('retrievals', 0)

    if retrievals < 3:
        sys_message = SystemMessage(content=DECIDE_RELEVANCE_PROMPT)
        response = model.bind_tools([retrieval_tool]).invoke([sys_message] + state['messages'])

        if len(response.tool_calls) == 0:
            tool_message = state['messages'][-1]
            sources = tool_message.artifact['sources']

            print('showing sources:', sources)

            response.content = f'{response.content}\n\nSources:\n\n{sources}'

    else:
        sys_message = SystemMessage(content=RETRIEVAL_FAILED_PROMPT)
        response = model.invoke([sys_message] + state['messages'])
        
    return {'messages': [response]}
