from typing import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langmem.short_term import RunningSummary

class State(MessagesState):
    context: dict[str, RunningSummary]

class HomeInputState(TypedDict):
    messages: list[AnyMessage]
    context: dict[str, RunningSummary]
