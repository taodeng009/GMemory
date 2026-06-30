import json
import unittest

from pydantic import ValidationError

from api.projector import ProjectorService
from api.schemas import ProjectorRequest


class FakeLLM:
    model_name = "fake-projector"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error:
            raise self.error
        return self.response


def make_request(raw_insights):
    return ProjectorRequest(
        goal="cool some bread and put it on countertop",
        subgoal=None,
        task_contract={"transformation": "cool"},
        raw_insights=raw_insights,
    )


def model_response(items):
    return json.dumps({"items": items})


class ProjectorServiceTests(unittest.TestCase):
    def test_empty_raw_insights_skips_llm(self):
        llm = FakeLLM(response="not used")
        response = ProjectorService(llm).project(make_request([]))

        self.assertEqual(response.bundle_status, "EMPTY")
        self.assertEqual(response.items, [])
        self.assertIsNone(response.error)
        self.assertEqual(llm.calls, [])

    def test_keep_rewrite_and_drop_preserve_order(self):
        raw = ["Keep me.", "Rewrite me.", "Drop me."]
        llm = FakeLLM(
            response=model_response(
                [
                    {"index": 0, "decision": "KEEP", "projected_insight": None},
                    {
                        "index": 1,
                        "decision": "REWRITE",
                        "projected_insight": "Use the current goal as the constraint.",
                    },
                    {"index": 2, "decision": "DROP", "projected_insight": None},
                ]
            )
        )

        response = ProjectorService(llm).project(make_request(raw))

        self.assertEqual(response.bundle_status, "HAS_CANDIDATES")
        self.assertIsNone(response.error)
        self.assertEqual([item.raw_insight for item in response.items], raw)
        self.assertEqual(
            [item.decision for item in response.items],
            ["KEEP", "REWRITE", "DROP"],
        )
        self.assertEqual(response.items[0].projected_insight, "Keep me.")
        self.assertEqual(
            response.items[1].projected_insight,
            "Use the current goal as the constraint.",
        )
        self.assertIsNone(response.items[2].projected_insight)
        self.assertEqual(llm.calls[0]["temperature"], 0.0)
        self.assertEqual(llm.calls[0]["num_comps"], 1)

    def test_all_drop_is_valid_empty_bundle_with_aligned_items(self):
        llm = FakeLLM(
            response=model_response(
                [
                    {"index": 0, "decision": "DROP", "projected_insight": None},
                    {"index": 1, "decision": "DROP", "projected_insight": None},
                ]
            )
        )

        response = ProjectorService(llm).project(make_request(["One", "Two"]))

        self.assertEqual(response.bundle_status, "EMPTY")
        self.assertEqual(len(response.items), 2)
        self.assertIsNone(response.error)

    def test_duplicate_candidate_drops_later_item(self):
        llm = FakeLLM(
            response=model_response(
                [
                    {"index": 0, "decision": "KEEP", "projected_insight": None},
                    {
                        "index": 1,
                        "decision": "REWRITE",
                        "projected_insight": "  SAME   INSIGHT  ",
                    },
                ]
            )
        )

        response = ProjectorService(llm).project(
            make_request(["Same insight", "Different raw text"])
        )

        self.assertEqual(response.items[0].decision, "KEEP")
        self.assertEqual(response.items[1].decision, "DROP")
        self.assertIsNone(response.items[1].projected_insight)
        self.assertEqual(response.items[1].risk_codes, ["DUPLICATE"])

    def test_timeout_fails_closed(self):
        llm = FakeLLM(error=TimeoutError("timed out"))

        response = ProjectorService(llm).project(make_request(["Do not expose me."]))

        self.assertEqual(response.bundle_status, "EMPTY")
        self.assertEqual(response.items, [])
        self.assertIn("TimeoutError", response.error)

    def test_invalid_json_fails_closed(self):
        response = ProjectorService(FakeLLM(response="not json")).project(
            make_request(["One"])
        )

        self.assertEqual(response.items, [])
        self.assertIn("invalid JSON", response.error)

    def test_wrong_item_count_fails_closed(self):
        response = ProjectorService(
            FakeLLM(
                response=model_response(
                    [{"index": 0, "decision": "KEEP", "projected_insight": None}]
                )
            )
        ).project(make_request(["One", "Two"]))

        self.assertEqual(response.items, [])
        self.assertIn("expected 2 items", response.error)

    def test_misaligned_indices_fail_closed(self):
        response = ProjectorService(
            FakeLLM(
                response=model_response(
                    [
                        {"index": 1, "decision": "KEEP", "projected_insight": None},
                        {"index": 0, "decision": "KEEP", "projected_insight": None},
                    ]
                )
            )
        ).project(make_request(["One", "Two"]))

        self.assertEqual(response.items, [])
        self.assertIn("expected item indices", response.error)

    def test_empty_rewrite_fails_closed(self):
        response = ProjectorService(
            FakeLLM(
                response=model_response(
                    [{"index": 0, "decision": "REWRITE", "projected_insight": " "}]
                )
            )
        ).project(make_request(["One"]))

        self.assertEqual(response.items, [])
        self.assertIn("empty projected_insight", response.error)

    def test_invalid_decision_fails_closed(self):
        response = ProjectorService(
            FakeLLM(
                response=model_response(
                    [{"index": 0, "decision": "MAYBE", "projected_insight": None}]
                )
            )
        ).project(make_request(["One"]))

        self.assertEqual(response.items, [])
        self.assertIn("schema validation", response.error)

    def test_request_rejects_empty_goal_and_insight(self):
        with self.assertRaises(ValidationError):
            ProjectorRequest(goal=" ", raw_insights=["One"])
        with self.assertRaises(ValidationError):
            ProjectorRequest(goal="Goal", raw_insights=[" "])


if __name__ == "__main__":
    unittest.main()
