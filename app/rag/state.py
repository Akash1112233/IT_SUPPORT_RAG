
from typing import Any, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description=(
            "Use kb for questions requiring company IT knowledge. "
            "Use direct only for greetings or casual conversation."
        )
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description=(
            "good means the supplied evidence directly supports "
            "a reliable answer. weak means evidence is insufficient."
        )
    )


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[dict[str, Any]]
