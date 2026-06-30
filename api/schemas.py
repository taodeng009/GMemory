from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


class EpisodeStep(BaseModel):
    subgoal: Optional[str] = None
    action: str
    observation: str
    reward: float = 0.0


class RetrieveRequest(BaseModel):
    task_type: str
    goal: str
    initial_observation: str
    max_chars: int = Field(default=4000, gt=0)
    render_mode: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRequest(BaseModel):
    task_type: str
    goal: str
    initial_observation: str
    success: StrictBool
    progress_rate: Optional[float] = None
    steps: list[EpisodeStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectorRequest(BaseModel):
    goal: str
    subgoal: None = None
    task_contract: dict[str, Any] = Field(default_factory=dict)
    raw_insights: list[str]

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal must not be empty")
        return value

    @field_validator("raw_insights")
    @classmethod
    def validate_raw_insights(cls, value: list[str]) -> list[str]:
        if any(not insight.strip() for insight in value):
            raise ValueError("raw insights must not contain empty strings")
        return value


ProjectorDecision = Literal["KEEP", "REWRITE", "DROP"]
ProjectorBundleStatus = Literal["HAS_CANDIDATES", "EMPTY"]


class ProjectorItem(BaseModel):
    raw_insight: str
    decision: ProjectorDecision
    projected_insight: Optional[str] = None
    applicable_phases: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    prohibited_assumptions: list[str] = Field(default_factory=list)
    risk_codes: list[str] = Field(default_factory=list)


class ProjectorResponse(BaseModel):
    bundle_status: ProjectorBundleStatus
    items: list[ProjectorItem] = Field(default_factory=list)
    error: Optional[str] = None


class MemoryStats(BaseModel):
    memory_size: int
    successful_count: int
    failed_count: int
    insight_count: int


class RetrieveResponse(BaseModel):
    memory_prompt: str
    stats: MemoryStats
    trace_id: str
    error: Optional[str] = None


class EpisodeResponse(BaseModel):
    stored: bool
    episode_id: Optional[str] = None
    trace_id: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    namespace: str
    memory_size: int
    error: Optional[str] = None


class TraceArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: dict[str, Any]
    derived: dict[str, Any]
    response: dict[str, Any]
    error: Optional[str] = None
