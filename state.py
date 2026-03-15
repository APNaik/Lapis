from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

# --- Pydantic Object for Constraints ---
class OutputConstraints(BaseModel):
    pages: Optional[int] = Field(default=1, description="Target number of pages")
    words_per_page: Optional[int] = Field(default=300, description="Approximate words per page")

class OutputFormat(BaseModel):
    name: str = "Standard Report"
    file_type: Literal["pdf", "doc"] = Field(..., description="The final export format")
    constraints: OutputConstraints

# --- The LangGraph State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_goal: str
    indexed_assets: Annotated[List[dict], operator.add]
    web_search_links: List[str]
    yt_vid_links: List[str]
    docs: List[str]
    draft: str
    output_format: OutputFormat
