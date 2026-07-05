import os
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv

from mas.memory.common import MASMessage

from .prompt_renderer import (
    count_because_lines,
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
from .semantic_gate import SemanticGateResult, SemanticGateService
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
    merge_enabled: bool = True
    merge_steps: int = 20
    merge_strategy: str = "original"
    atomic_merge_ratio: float = 0.33
    atomic_merge_max_words: int = 30
    strip_alfworld_prefix_for_retrieval: bool = False
    render_mode: str = "default"
    insight_style: str = "original"
    semantic_gate_version: str = "none"


class GMemoryApiService:
    def __init__(
        self,
        config: Optional[GMemoryApiConfig] = None,
        tracer: Optional[ApiTracer] = None,
        semantic_gate: Optional[SemanticGateService] = None,
    ):
        self.config = config or GMemoryApiConfig()
        self._load_env_config()
        self.tracer = tracer or ApiTracer()
        self._semantic_gate = semantic_gate
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
            "insight_style": self.config.insight_style,
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
                derived["because_line_count_before"] = count_because_lines(insights)
                rendered_insights = insights
                gate_error = None
                gate_enabled = self.config.semantic_gate_version != "none"
                if gate_enabled and render_mode in {"default", "insight_only"}:
                    gate_result = self._run_semantic_gate(
                        request.goal,
                        request.initial_observation,
                        insights,
                    )
                    rendered_insights = gate_result.passed_insights
                    gate_error = gate_result.error
                    derived["semantic_gate"] = self._semantic_gate_trace(
                        gate_result,
                        raw_insight_count=len(insights),
                    )
                else:
                    derived["semantic_gate"] = {
                        "enabled": gate_enabled,
                        "applied": False,
                        "version": self.config.semantic_gate_version,
                    }

                memory_prompt = self._render_memory_prompt(
                    success,
                    rendered_insights,
                    task_description,
                    render_mode,
                )
                derived["because_line_count_after"] = count_because_lines(
                    self._normalize_rendered_insights(rendered_insights, render_mode)
                )
                memory_prompt = memory_prompt[: request.max_chars]
                stats = MemoryStats(
                    memory_size=memory_size,
                    successful_count=len(success),
                    failed_count=len(failed),
                    insight_count=len(insights),
                )
                if not memory_prompt:
                    error = (
                        f"semantic gate failed: {gate_error}"
                        if gate_error
                        else "no retrieval result"
                    )
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
                global_config={
                    "working_dir": self.config.working_dir,
                    "hop": self.config.hop,
                    "merge_enabled": self.config.merge_enabled,
                    "merge_steps": self.config.merge_steps,
                    "merge_strategy": self.config.merge_strategy,
                    "atomic_merge_ratio": self.config.atomic_merge_ratio,
                    "atomic_merge_max_words": self.config.atomic_merge_max_words,
                },
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
            return render_insight_only_memory_prompt(insights, self.config.insight_style)
        return render_memory_prompt(successful, insights, task_description)

    def _normalize_rendered_insights(self, insights: list[str], render_mode: str) -> list[str]:
        if render_mode != "insight_only":
            return insights
        from .prompt_renderer import normalize_insight_text

        return [normalize_insight_text(insight, self.config.insight_style) for insight in insights]

    def _resolve_render_mode(self, request_render_mode: Optional[str]) -> str:
        render_mode = request_render_mode or self.config.render_mode
        render_mode = str(render_mode or "default").strip().lower()
        if render_mode in {"key_steps_only", "goal_key_steps_only", "insight_only"}:
            return render_mode
        return "default"

    def _run_semantic_gate(
        self,
        goal: str,
        initial_observation: str,
        insights: list[str],
    ) -> SemanticGateResult:
        try:
            return self._get_semantic_gate().filter(goal, initial_observation, insights)
        except Exception as exc:
            return SemanticGateResult(
                passed_insights=[],
                items=[],
                error=self._summarize_error(exc),
            )

    def _get_semantic_gate(self) -> SemanticGateService:
        if self._semantic_gate is None:
            from mas.llm import GPTChat

            self._semantic_gate = SemanticGateService(
                llm_client=GPTChat(model_name=self.config.llm_model),
                version=self.config.semantic_gate_version,
            )
        return self._semantic_gate

    def _semantic_gate_trace(
        self,
        result: SemanticGateResult,
        raw_insight_count: int,
    ) -> dict:
        pass_count = sum(1 for item in result.items if item.decision == "PASS")
        block_count = (
            raw_insight_count - pass_count
            if result.error
            else sum(1 for item in result.items if item.decision == "BLOCK")
        )
        llm_client = getattr(self._semantic_gate, "llm_client", None)
        prompt_version = getattr(
            self._semantic_gate,
            "prompt_version",
            f"api-semantic-gate-{self.config.semantic_gate_version}",
        )
        return {
            "enabled": True,
            "applied": True,
            "version": self.config.semantic_gate_version,
            "prompt_version": prompt_version,
            "model": getattr(llm_client, "model_name", None),
            "temperature": 0.0,
            "raw_insight_count": raw_insight_count,
            "pass_count": pass_count,
            "block_count": block_count,
            "items": [item.model_dump() for item in result.items],
            "raw_model_output": result.raw_model_output,
            "error": result.error,
        }

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
        self.config.merge_enabled = self._resolve_merge_enabled(
            os.getenv("GMEMORY_API_MERGE", "enabled" if self.config.merge_enabled else "disabled")
        )
        self.config.merge_steps = self._env_positive_int(
            "GMEMORY_API_MERGE_STEPS", self.config.merge_steps
        )
        self.config.merge_strategy = self._resolve_merge_strategy(
            os.getenv("GMEMORY_API_MERGE_STRATEGY", self.config.merge_strategy)
        )
        self.config.atomic_merge_ratio = self._env_ratio(
            "GMEMORY_API_ATOMIC_MERGE_RATIO", self.config.atomic_merge_ratio
        )
        self.config.atomic_merge_max_words = self._env_positive_int(
            "GMEMORY_API_ATOMIC_MERGE_MAX_WORDS", self.config.atomic_merge_max_words
        )
        self.config.render_mode = os.getenv("GMEMORY_API_RENDER_MODE", self.config.render_mode)
        self.config.insight_style = self._resolve_insight_style(
            os.getenv("GMEMORY_API_INSIGHT_STYLE", self.config.insight_style)
        )
        self.config.semantic_gate_version = self._resolve_semantic_gate_version(
            os.getenv(
                "GMEMORY_API_SEMANTIC_GATE_VERSION",
                self.config.semantic_gate_version,
            )
        )
        self.config.strip_alfworld_prefix_for_retrieval = self._env_bool(
            "GMEMORY_API_STRIP_ALFWORLD_PREFIX_FOR_RETRIEVAL",
            self.config.strip_alfworld_prefix_for_retrieval,
        )

    def _resolve_insight_style(self, insight_style: str) -> str:
        insight_style = str(insight_style or "original").strip().lower()
        if insight_style in {"original", "no_because"}:
            return insight_style
        return "original"

    def _resolve_semantic_gate_version(self, version: str) -> str:
        version = str(version or "none").strip().lower()
        if version in {"none", "v1", "v2", "v3", "v4", "v5"}:
            return version
        return "none"

    def _resolve_merge_enabled(self, value: str) -> bool:
        value = str(value or "enabled").strip().lower()
        if value == "disabled":
            return False
        return True

    def _resolve_merge_strategy(self, value: str) -> str:
        value = str(value or "original").strip().lower()
        if value in {"original", "atomic_v1"}:
            return value
        return "original"

    def _env_ratio(self, name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if 0 < parsed <= 1 else default

    def _env_positive_int(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _env_bool(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
