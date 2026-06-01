import os
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv

from mas.memory.common import MASMessage

from .prompt_renderer import render_memory_prompt
from .schemas import (
    EpisodeRequest,
    EpisodeResponse,
    MemoryStats,
    RetrieveRequest,
    RetrieveResponse,
)
from .tracing import ApiTracer


@dataclass
class GMemoryApiConfig:
    namespace: str = "hiagent-cross-task"
    working_dir: str = "./.db/hiagent_gmemory_api"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo-0125"
    successful_topk: int = 1
    failed_topk: int = 0
    insights_topk: int = 3
    threshold: float = 0.0
    hop: int = 1


class GMemoryApiService:
    def __init__(self, config: Optional[GMemoryApiConfig] = None, tracer: Optional[ApiTracer] = None):
        self.config = config or GMemoryApiConfig()
        self.tracer = tracer or ApiTracer()
        self._memory = None
        self._init_error = None

    @property
    def namespace(self) -> str:
        return self.config.namespace

    def health(self) -> dict:
        try:
            memory_size = self.memory_size
            return {
                "ok": self._init_error is None,
                "backend": "g-memory",
                "namespace": self.namespace,
                "memory_size": memory_size,
                "error": self._init_error,
            }
        except Exception as exc:
            return {
                "ok": False,
                "backend": "g-memory",
                "namespace": self.namespace,
                "memory_size": 0,
                "error": self._summarize_error(exc),
            }

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        trace_id = self.tracer.new_trace_id()
        request_dict = request.model_dump()
        task_main, task_description, task_main_rule = self._derive_task_fields(
            request.task_type,
            request.goal,
            request.initial_observation,
            request.metadata,
        )
        derived = {
            "query_task": task_main,
            "task_main_rule": task_main_rule,
            "task_description": task_description,
        }

        error = None
        memory_prompt = ""
        stats = self._empty_stats()

        try:
            memory_size = self.memory_size
            stats.memory_size = memory_size
            if memory_size == 0:
                error = "empty memory"
            else:
                success, failed, insights = self._memory.retrieve_memory(
                    query_task=task_main,
                    successful_topk=self.config.successful_topk,
                    failed_topk=self.config.failed_topk,
                    insight_topk=self.config.insights_topk,
                    threshold=self.config.threshold,
                )
                memory_prompt = self._render_memory_prompt(success, insights, task_description)
                memory_prompt = memory_prompt[: request.max_chars]
                stats = MemoryStats(
                    memory_size=memory_size,
                    successful_count=len(success),
                    failed_count=len(failed),
                    insight_count=len(insights),
                )
                if not memory_prompt:
                    error = "no retrieval result"
        except Exception as exc:
            error = self._summarize_error(exc)
            stats = self._safe_stats()
            memory_prompt = ""

        response = RetrieveResponse(
            memory_prompt=memory_prompt,
            stats=stats,
            trace_id=trace_id,
            error=error,
        )
        self.tracer.record(trace_id, "/retrieve", request_dict, derived, response.model_dump(), error)
        return response

    def save_episode(self, request: EpisodeRequest) -> EpisodeResponse:
        trace_id = self.tracer.new_trace_id()
        request_dict = request.model_dump()
        task_main, task_description, task_main_rule = self._derive_task_fields(
            request.task_type,
            request.goal,
            request.initial_observation,
            request.metadata,
        )
        label = request.success
        mas_message = MASMessage(task_main=task_main, task_description=task_description, label=label)
        mas_message.add_extra_field("task_type", request.task_type)
        mas_message.add_extra_field("metadata", request.metadata)
        if request.progress_rate is not None:
            mas_message.add_extra_field("progress_rate", request.progress_rate)

        for step in request.steps:
            if step.subgoal is not None:
                mas_message.add_extra_field("last_subgoal", step.subgoal)
            mas_message.move_state(step.action, step.observation, reward=step.reward)

        derived = {
            "task_main": task_main,
            "task_description": task_description,
            "task_main_rule": task_main_rule,
            "label": label,
            "step_count": len(request.steps),
        }

        try:
            _ = self.memory_size
            self._memory.add_memory(mas_message)
            response = EpisodeResponse(stored=True, episode_id=uuid4().hex, trace_id=trace_id)
            error = None
        except Exception as exc:
            error = self._summarize_error(exc)
            response = EpisodeResponse(stored=False, episode_id=None, trace_id=trace_id, error=error)

        self.tracer.record(trace_id, "/episodes", request_dict, derived, response.model_dump(), error)
        return response

    @property
    def memory_size(self) -> int:
        try:
            return int(self._memory.memory_size)
        except Exception:
            if self._memory is None:
                self._build_memory()
                return int(self._memory.memory_size)
            raise

    def _build_memory(self) -> None:
        load_dotenv()

        from mas.llm import GPTChat
        from mas.memory.mas_memory.GMemory import GMemory
        from mas.utils import EmbeddingFunc

        self.config.llm_model = os.getenv("GMEMORY_API_MODEL", self.config.llm_model)
        self.config.working_dir = os.getenv("GMEMORY_API_WORKING_DIR", self.config.working_dir)
        self.config.namespace = os.getenv("GMEMORY_API_NAMESPACE", self.config.namespace)
        self.config.embedding_model = os.getenv("GMEMORY_API_EMBEDDING_MODEL", self.config.embedding_model)

        try:
            os.makedirs(self.config.working_dir, exist_ok=True)
            self._memory = GMemory(
                namespace=self.config.namespace,
                global_config={"working_dir": self.config.working_dir, "hop": self.config.hop},
                llm_model=GPTChat(model_name=self.config.llm_model),
                embedding_func=EmbeddingFunc(self.config.embedding_model),
            )
            self._init_error = None
        except Exception as exc:
            self._init_error = self._summarize_error(exc)
            raise

    def _derive_task_fields(
        self,
        task_type: str,
        goal: str,
        initial_observation: str,
        metadata: dict,
    ) -> tuple[str, str, str]:
        normalized_type = task_type.lower()
        metadata_env = str(metadata.get("env", "")).lower()
        if normalized_type.startswith("alfworld") or metadata_env == "alfworld":
            task_main = f"alfworld-{goal}"
            rule = "alfworld-prefix-goal"
        else:
            task_main = goal
            rule = "pddl-goal"
        task_description = f"Here is your initial observation: {initial_observation}\n**Here is your task: {goal}"
        return task_main, task_description, rule

    def _render_memory_prompt(self, successful: list[MASMessage], insights: list[str], task_description: str) -> str:
        return render_memory_prompt(successful, insights, task_description)

    def _empty_stats(self) -> MemoryStats:
        return MemoryStats(memory_size=0, successful_count=0, failed_count=0, insight_count=0)

    def _safe_stats(self) -> MemoryStats:
        try:
            memory_size = self.memory_size
        except Exception:
            memory_size = 0
        return MemoryStats(memory_size=memory_size, successful_count=0, failed_count=0, insight_count=0)

    def _summarize_error(self, exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {str(exc)[:500]}"
