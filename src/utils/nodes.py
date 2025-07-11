from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from src.utils.state import State
from src.utils.tools import retrieval_tool, cheatsheet_tool

max_retrievals = 3
model = AzureChatOpenAI(azure_deployment='gpt-4o', api_version='2024-10-21')

# ========== Prompts and messages ==========

COURSE_PROMPT = "You are a virtual teaching assistant and your job is to help students by answering relevant, theoretical questions."

DECIDE_RETRIEVAL_PROMPT = f"""{COURSE_PROMPT}
If the student questions should be course-related.
If the conversation holds the answer to the student's question, use it.
Use {retrieval_tool.name} to get information to the student's questions.
Use {cheatsheet_tool.name} to generate a cheatsheet if requested."""

DECIDE_RELEVANCE_PROMPT = f"""{COURSE_PROMPT}
If the results provided by the tool have the answer to the user's query, just say 'satisfactory.'
If not, refine your query based and use the tool {retrieval_tool.name} again."""

RETRIEVAL_FAILED_PROMPT = f"""{COURSE_PROMPT}
If the results provided by the tool have the answer to the user's query, generate an answer based on these results.
If not, state that you were unable to get the required information and ask the user to refine their question."""

FAILURE_MESSAGE = 'Retrieval failed. Please ask the user to refine their question.'

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
    tools = [retrieval_tool, cheatsheet_tool]
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

        print('Retrieval satisfactory: ', getattr(response, "tool_calls", None))
        # Return the unmodified state if no tools were called since it carries the outputs
        return state if getattr(response, "tool_calls", None) else {'messages': [response]}
    
    else:
        # Signal tool failure and recommended action
        response = ToolMessage(content=FAILURE_MESSAGE)
        return {'messages': [response]}