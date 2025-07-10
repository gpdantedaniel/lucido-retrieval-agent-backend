from langgraph.graph import MessagesState

class State(MessagesState):
    retrievals: int = 0
