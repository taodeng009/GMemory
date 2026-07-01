import sys
import types
import unittest
from unittest.mock import Mock


try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

memory_package = types.ModuleType("mas.memory")
memory_package.__path__ = []
memory_common = types.ModuleType("mas.memory.common")
memory_common.MASMessage = type("MASMessage", (), {})
sys.modules.setdefault("mas.memory", memory_package)
sys.modules.setdefault("mas.memory.common", memory_common)

from api.schemas import RetrieveRequest
from api.semantic_gate import (
    SemanticGateItem,
    SemanticGateResult,
)
from api.service import GMemoryApiConfig, GMemoryApiService


class FakeTracer:
    def __init__(self):
        self.records = []

    def new_trace_id(self):
        return "trace-id"

    def record(self, trace_id, endpoint, request, derived, response, error=None):
        self.records.append(
            {
                "trace_id": trace_id,
                "endpoint": endpoint,
                "request": request,
                "derived": derived,
                "response": response,
                "error": error,
            }
        )


class FakeMemory:
    memory_size = 5

    def __init__(self, successful=None, failed=None, insights=None):
        self.successful = successful or []
        self.failed = failed or []
        self.insights = insights or []
        self.calls = []

    def retrieve_memory(self, **kwargs):
        self.calls.append(kwargs)
        return self.successful, self.failed, self.insights


class FakeGate:
    llm_client = type("LLM", (), {"model_name": "fake-gate"})()

    def __init__(self, result):
        self.result = result
        self.calls = []

    def filter(self, goal, initial_observation, raw_insights):
        self.calls.append(
            {
                "goal": goal,
                "initial_observation": initial_observation,
                "raw_insights": raw_insights,
            }
        )
        return self.result


def request(render_mode="insight_only"):
    return RetrieveRequest(
        task_type="alfworld",
        goal="cool some bread and put it on countertop",
        initial_observation="A countertop is visible.",
        render_mode=render_mode,
    )


def build_service(memory, gate, version="v2"):
    config = GMemoryApiConfig(
        render_mode="insight_only",
        insight_style="original",
        semantic_gate_version=version,
    )
    tracer = FakeTracer()
    service = GMemoryApiService(config=config, tracer=tracer, semantic_gate=gate)
    service.config.semantic_gate_version = version
    service.config.render_mode = "insight_only"
    service.config.insight_style = "original"
    service._memory = memory
    return service, tracer


class SemanticGateRetrieveTests(unittest.TestCase):
    def test_retrieve_renders_only_passed_insights_and_preserves_raw_stats(self):
        raw = ["Pass exactly.", "Block exactly."]
        gate = FakeGate(
            SemanticGateResult(
                passed_insights=[raw[0]],
                items=[
                    SemanticGateItem(index=0, decision="PASS"),
                    SemanticGateItem(index=1, decision="BLOCK"),
                ],
            )
        )
        memory = FakeMemory(insights=raw)
        service, tracer = build_service(memory, gate)

        response = service.retrieve(request())

        self.assertIn(raw[0], response.memory_prompt)
        self.assertNotIn(raw[1], response.memory_prompt)
        self.assertEqual(response.stats.insight_count, 2)
        self.assertIsNone(response.error)
        self.assertEqual(gate.calls[0]["goal"], request().goal)
        self.assertEqual(gate.calls[0]["initial_observation"], request().initial_observation)
        self.assertEqual(
            memory.calls[0],
            {
                "query_task": f"alfworld-{request().goal}",
                "successful_topk": service.config.successful_topk,
                "failed_topk": service.config.failed_topk,
                "insight_topk": service.config.insights_topk,
                "threshold": service.config.threshold,
            },
        )
        gate_trace = tracer.records[0]["derived"]["semantic_gate"]
        self.assertEqual(gate_trace["raw_insight_count"], 2)
        self.assertEqual(gate_trace["pass_count"], 1)
        self.assertEqual(gate_trace["block_count"], 1)
        self.assertEqual(gate_trace["version"], "v2")
        self.assertEqual(gate_trace["prompt_version"], "api-semantic-gate-v2")

    def test_gate_version_resolution_is_strict(self):
        service, _ = build_service(FakeMemory(), None, version="none")

        self.assertEqual(service._resolve_semantic_gate_version("V1"), "v1")
        self.assertEqual(service._resolve_semantic_gate_version(" v2 "), "v2")
        self.assertEqual(service._resolve_semantic_gate_version("V3"), "v3")
        self.assertEqual(service._resolve_semantic_gate_version("disabled"), "none")

    def test_disabled_gate_preserves_original_behavior(self):
        raw = ["One.", "Two."]
        gate = FakeGate(
            SemanticGateResult(passed_insights=[], items=[], error="should not run")
        )
        service, tracer = build_service(FakeMemory(insights=raw), gate, version="none")

        response = service.retrieve(request())

        self.assertIn(raw[0], response.memory_prompt)
        self.assertIn(raw[1], response.memory_prompt)
        self.assertEqual(gate.calls, [])
        self.assertEqual(
            tracer.records[0]["derived"]["semantic_gate"],
            {"enabled": False, "applied": False, "version": "none"},
        )

    def test_gate_failure_keeps_other_memory_without_response_error(self):
        raw = ["Unsafe raw insight."]
        gate = FakeGate(
            SemanticGateResult(
                passed_insights=[],
                items=[],
                error="TimeoutError: timed out",
            )
        )
        service, tracer = build_service(
            FakeMemory(successful=[object()], insights=raw),
            gate,
        )
        service._render_memory_prompt = Mock(return_value="successful memory")

        response = service.retrieve(request(render_mode="default"))

        self.assertEqual(response.memory_prompt, "successful memory")
        self.assertIsNone(response.error)
        self.assertEqual(response.stats.insight_count, 1)
        self.assertEqual(
            tracer.records[0]["derived"]["semantic_gate"]["error"],
            "TimeoutError: timed out",
        )

    def test_gate_failure_sets_response_error_only_when_prompt_is_empty(self):
        gate = FakeGate(
            SemanticGateResult(
                passed_insights=[],
                items=[],
                error="ValueError: invalid JSON",
            )
        )
        service, _ = build_service(FakeMemory(insights=["unsafe"]), gate)

        response = service.retrieve(request())

        self.assertEqual(response.memory_prompt, "")
        self.assertEqual(
            response.error,
            "semantic gate failed: ValueError: invalid JSON",
        )

    def test_all_block_is_not_a_gate_error(self):
        gate = FakeGate(
            SemanticGateResult(
                passed_insights=[],
                items=[SemanticGateItem(index=0, decision="BLOCK")],
            )
        )
        service, _ = build_service(FakeMemory(insights=["blocked"]), gate)

        response = service.retrieve(request())

        self.assertEqual(response.memory_prompt, "")
        self.assertEqual(response.error, "no retrieval result")

    def test_key_steps_modes_skip_gate(self):
        gate = FakeGate(
            SemanticGateResult(passed_insights=[], items=[], error="should not run")
        )
        service, tracer = build_service(FakeMemory(insights=["raw"]), gate)
        service._render_memory_prompt = Mock(return_value="key steps")

        response = service.retrieve(request(render_mode="key_steps_only"))

        self.assertEqual(response.memory_prompt, "key steps")
        self.assertEqual(gate.calls, [])
        self.assertEqual(
            tracer.records[0]["derived"]["semantic_gate"],
            {"enabled": True, "applied": False, "version": "v2"},
        )


if __name__ == "__main__":
    unittest.main()
