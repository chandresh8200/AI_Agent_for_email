from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    """A model for a single tool call in the plan."""
    tool_name: str
    tool_kwargs: Dict[str, Any]

class AgentState(BaseModel):
    """
    Represents the formal memory or 'state' of our agent.
    """
    # This field will be used to pass in text directly, bypassing the microphone.
    initialUserInput: Optional[str] = None

    # The rest of the state remains the same
    userInput: str = ""
    correctedInput: str = ""
    plan: List[ToolCall] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: str = ""
    error: Optional[str] = None
    needs_clarification: bool = False
    clarification_question: str = ""

    class Config:
        arbitrary_types_allowed = True

