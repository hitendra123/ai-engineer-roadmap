from langgraph.graph import MessagesState
from typing import Optional

class SupportState(MessagesState):
    user_name: Optional[str] = None
    intent: Optional[str] = None
    turn_count: int = 0
    summary: Optional[str] = None