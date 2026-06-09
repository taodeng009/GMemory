import os
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv

from mas.memory.common import MASMessage

from .prompt_renderer import (
    render_goal_key_steps_only_memory_prompt,
    render_insight_only_memory_prompt,
    render_key_steps_only_memory_prompt,
    render_memory_prompt,
)
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
    strip_alfworld_prefix_for_retrieval: bool = False
    render_mode: str = "default"


class GMemoryApiService:
    def __init__(self, config: Optional[GMemoryApiConfig] = None, tracer: Optional[ApiTracer] = None):
        self.config = config or GMemoryApiConfig()
        self._load_env_config()
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
        task_main, task_description, task_main_rule, raw_task_main = self._derive_task_fields(
            request.task_type,
            request.goal,
            request.initial_observation,
            request.metadata,
        )
        render_mode = self._resolve_render_mode(request.render_mode)
        derived = {
            "query_task": task_main,
            "raw_query_task": raw_task_main,
            "task_main_rule": task_main_rule,
            "task_description": task_description,
            "render_mode": render_mode,
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
                retrieval_debug = getattr(self._memory, "last_retrieval_debug", None)
                if retrieval_debug:
                    derived["retrieval_debug"] = retrieval_debug
                memory_prompt = self._render_memory_prompt(success, insights, task_description, render_mode)
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
        task_main, task_description, task_main_rule, raw_task_main = self._derive_task_fields(
            request.task_type,
            request.goal,
            request.initial_observation,
            request.metadata,
        )
        label = request.success
        mas_message = MASMessage(task_main=task_main, task_description=task_description, label=label)
        mas_message.add_extra_field("task_type", request.task_type)
        metadata = dict(request.metadata)
        if raw_task_main != task_main:
            metadata["raw_task_main"] = raw_task_main
        mas_message.add_extra_field("metadata", metadata)
        if request.progress_rate is not None:
            mas_message.add_extra_field("progress_rate", request.progress_rate)

        for step in request.steps:
            if step.subgoal is not None:
                mas_message.add_extra_field("last_subgoal", step.subgoal)
            mas_message.move_state(step.action, step.observation, reward=step.reward)

        derived = {
            "task_main": task_main,
            "raw_task_main": raw_task_main,
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

        self._load_env_config()

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
    ) -> tuple[str, str, str, str]:
        normalized_type = task_type.lower()
        metadata_env = str(metadata.get("env", "")).lower()
        if normalized_type.startswith("alfworld") or metadata_env == "alfworld":
            raw_task_main = f"alfworld-{goal}"
            if self.config.strip_alfworld_prefix_for_retrieval:
                task_main = goal
                rule = "alfworld-prefix-stripped"
            else:
                task_main = raw_task_main
                rule = "alfworld-prefix-goal"
        else:
            task_main = goal
            raw_task_main = task_main
            rule = "pddl-goal"
        task_description = f"Here is your initial observation: {initial_observation}\n**Here is your task: {goal}"
        return task_main, task_description, rule, raw_task_main

    def _render_memory_prompt(
        self,
        successful: list[MASMessage],
        insights: list[str],
        task_description: str,
        render_mode: str,
    ) -> str:
        if render_mode == "key_steps_only":
            return render_key_steps_only_memory_prompt(successful)
        if render_mode == "goal_key_steps_only":
            return render_goal_key_steps_only_memory_prompt(successful)
        if render_mode == "insight_only":
            return render_insight_only_memory_prompt(insights)
        return render_memory_prompt(successful, insights, task_description)

    def _resolve_render_mode(self, request_render_mode: Optional[str]) -> str:
        render_mode = request_render_mode or self.config.render_mode
        render_mode = str(render_mode or "default").strip().lower()
        if render_mode in {"key_steps_only", "goal_key_steps_only", "insight_only"}:
            return render_mode
        return "default"

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

    def _load_env_config(self) -> None:
        load_dotenv()
        self.config.llm_model = os.getenv("GMEMORY_API_MODEL", self.config.llm_model)
        self.config.working_dir = os.getenv("GMEMORY_API_WORKING_DIR", self.config.working_dir)
        self.config.namespace = os.getenv("GMEMORY_API_NAMESPACE", self.config.namespace)
        self.config.embedding_model = os.getenv("GMEMORY_API_EMBEDDING_MODEL", self.config.embedding_model)
        self.config.successful_topk = int(os.getenv("GMEMORY_API_SUCCESSFUL_TOPK", self.config.successful_topk))
        self.config.failed_topk = int(os.getenv("GMEMORY_API_FAILED_TOPK", self.config.failed_topk))
        self.config.insights_topk = int(os.getenv("GMEMORY_API_INSIGHTS_TOPK", self.config.insights_topk))
        self.config.threshold = float(os.getenv("GMEMORY_API_THRESHOLD", self.config.threshold))
        self.config.hop = int(os.getenv("GMEMORY_API_HOP", self.config.hop))
        self.config.render_mode = os.getenv("GMEMORY_API_RENDER_MODE", self.config.render_mode)
        self.config.strip_alfworld_prefix_for_retrieval = self._env_bool(
            "GMEMORY_API_STRIP_ALFWORLD_PREFIX_FOR_RETRIEVAL",
            self.config.strip_alfworld_prefix_for_retrieval,
        )

    def _env_bool(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
