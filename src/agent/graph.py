from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.utils.state import State
from src.utils.nodes import retrieval_needed, judge_results, init_defaults
from src.utils.tools import retrieval_tool, cheatsheet_tool

workflow = StateGraph(State)

def retrieval_finished(state: State) -> bool:
    """
    Determines whether retrieval is finished by
    the presence of a tool call in the response
    """
    last = state['messages'][-1]
    return not last.tool_calls

# Add our nodes
workflow.add_node('init', init_defaults)
workflow.add_node('retrieval_needed', retrieval_needed)
workflow.add_node('retrieval_tool', ToolNode([retrieval_tool]))
workflow.add_node(judge_results)

# Add our edges
workflow.add_edge(START, 'init')
workflow.add_edge('init', 'retrieval_needed')
workflow.add_conditional_edges('retrieval_needed', tools_condition, {'tools': 'retrieval_tool', END: END})
workflow.add_edge('retrieval_tool', 'judge_results')
workflow.add_conditional_edges('judge_results', retrieval_finished, {True: 'retrieval_needed', False: 'retrieval_tool'})

graph = workflow.compile()