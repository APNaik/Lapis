from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator


class OutputConstraints(BaseModel):
    pages: Optional[int] = Field(default=1, description="Target number of pages")
    words_per_page: Optional[int] = Field(default=300, description="Approximate words per page")


class OutputFormat(BaseModel):
    name: str = "Standard Report"
    file_type: Literal["pdf"] = Field(default="pdf", description="The final export format")
    constraints: OutputConstraints = Field(default_factory=OutputConstraints)


class LabState(TypedDict, total=False):
    lab_id: str
    lab_title: str
    research_goal: str
    status: Literal[
        "created",
        "ingesting_seed",
        "researching",
        "needs_input",
        "drafting",
        "complete",
        "failed",
    ]
    status_message: str
    seed_sources: Annotated[List[dict], operator.add]
    indexed_assets: Annotated[List[dict], operator.add]
    discovered_sources: Annotated[List[dict], operator.add]
    research_notes: Annotated[List[str], operator.add]
    requested_resources: Annotated[List[str], operator.add]
    draft: str
    output_format: OutputFormat
    report_path: str
    confidence: float
    error: str


# Backwards-compatible alias for older imports during migration.
AgentState = LabState
