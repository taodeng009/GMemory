import json
import os
import sys
import types
import unittest

os.environ["OPENAI_API_BASE"] = "http://localhost"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["GMEMORY_API_MODEL"] = "test-model"

memory_package = types.ModuleType("mas.memory")
memory_package.__path__ = []
memory_common = types.ModuleType("mas.memory.common")
memory_common.MASMessage = type("MASMessage", (), {})
sys.modules.setdefault("mas.memory", memory_package)
sys.modules.setdefault("mas.memory.common", memory_common)

from fastapi.testclient import TestClient

from api import server


class FakeLLM:
    model_name = "fake-projector"

    def __call__(self, messages, **kwargs):
        return json.dumps(
            {
                "items": [
                    {
                        "index": 0,
                        "decision": "KEEP",
                        "projected_insight": None,
                    }
                ]
            }
        )


class ProjectorEndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_llm_client = server.projector_service.llm_client
        self.original_tracer = server.projector_service.tracer
        server.projector_service.llm_client = FakeLLM()
        server.projector_service.tracer = None
        self.client = TestClient(server.app)

    def tearDown(self):
        server.projector_service.llm_client = self.original_llm_client
        server.projector_service.tracer = self.original_tracer

    def test_project_endpoint_uses_projector_service(self):
        response = self.client.post(
            "/api/v1/memory/project",
            json={
                "goal": "cool some bread and put it on countertop",
                "subgoal": None,
                "task_contract": {},
                "raw_insights": ["Check all preconditions before acting."],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["bundle_status"], "HAS_CANDIDATES")
        self.assertIsNone(payload["error"])
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["decision"], "KEEP")
        self.assertEqual(
            payload["items"][0]["projected_insight"],
            "Check all preconditions before acting.",
        )


if __name__ == "__main__":
    unittest.main()
