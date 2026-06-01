from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRequest(BaseModel):
    task_type: str
    goal: str
    initial_observation: str
    success: StrictBool
    progress_rate: Optional[float] = None
    steps: list[EpisodeStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
