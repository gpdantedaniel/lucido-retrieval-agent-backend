from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.utils.state import State
from src.utils.nodes import home, init_defaults
from src.utils.tools import retrieval_tool, cheatsheet_tool, quiz_tool

workflow = StateGraph(State)

def retrieval_finished(state: State) -> bool:
    """
    Determines whether retrieval is finished by
    the presence of a tool call in the response
    """
    last = state['messages'][-1]
    return not last.tool_calls

def route_tool(state: State) -> str:
    """
    Returns the name of the tool to route
    to depending on the last message
    """
    last = state['messages'][-1]

    if hasattr(last, 'tool_calls') and len(last.tool_calls) > 0:
        return last.tool_calls[0]['name']
    
    return END

tool_routes = {
    'retrieval_tool': 'retrieval_tool', 
    'cheatsheet_tool': 'cheatsheet_tool',
    'quiz_tool': 'quiz_tool',
    END: END
}

# Add our nodes
workflow.add_node('init', init_defaults)
workflow.add_node('home', home)
workflow.add_node('retrieval_tool', ToolNode([retrieval_tool]))
workflow.add_node('cheatsheet_tool', ToolNode([cheatsheet_tool]))
workflow.add_node('quiz_tool', ToolNode([quiz_tool]))

# Add our edges
workflow.add_edge(START, 'init')
workflow.add_edge('init', 'home')
workflow.add_conditional_edges('home', route_tool, tool_routes)
workflow.add_edge('retrieval_tool', 'home')
workflow.add_edge('cheatsheet_tool', 'home')
workflow.add_edge('quiz_tool', 'home')

graph = workflow.compile()