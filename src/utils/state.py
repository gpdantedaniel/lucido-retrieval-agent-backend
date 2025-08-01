from typing import TypedDict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

class State(MessagesState):
    context: dict[str, Any]

class HomeInputState(TypedDict):
    messages: list[AnyMessage]
    context: dict[str, Any]
