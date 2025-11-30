"""State definitions for the DocMana LangGraph flow."""

from typing import Annotated, Dict, List, Optional

from langgraph.graph import MessagesState


# Reducers for merging state across nodes.
def override(_old, new):
    return new


def merge_dicts(old: Dict[str, str], new: Dict[str, str]):
    merged = dict(old)
    merged.update(new or {})
    return merged


class GraphState(MessagesState):
    """Graph state shared across nodes."""

    user_query: str
    sub_questions: Annotated[List[str], override]
    tool_results: Annotated[Dict[str, str], merge_dicts]
    final_answer: Annotated[Optional[str], override]
    error: Annotated[Optional[str], override]
