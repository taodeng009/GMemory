import json
import tempfile
import unittest
from pathlib import Path

from api.projector import ProjectorService
from scripts.projector_file_runner import run_file


class SequencedFakeLLM:
    model_name = "fake-projector"

    def __init__(self):
        self.call_count = 0

    def __call__(self, messages, **kwargs):
        self.call_count += 1
        return json.dumps(
            {
                "items": [
                    {
                        "index": 0,
                        "decision": "REWRITE",
                        "projected_insight": f"Projected {self.call_count}",
                    }
                ]
            }
        )


def request_line(goal, insight):
    return json.dumps(
        {
            "goal": goal,
            "subgoal": None,
            "task_contract": {},
            "raw_insights": [insight],
        },
        ensure_ascii=False,
    )


class ProjectorFileRunnerTests(unittest.TestCase):
    def test_utf8_multiple_lines_and_invalid_middle_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "输入.jsonl"
            output_path = Path(temp_dir) / "输出.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        request_line("冷却面包", "检查当前状态。"),
                        "{invalid json",
                        request_line("放置苹果", "遵循当前目标。"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            llm = SequencedFakeLLM()

            run_file(input_path, output_path, ProjectorService(llm))

            results = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["items"][0]["raw_insight"], "检查当前状态。")
            self.assertEqual(results[0]["items"][0]["projected_insight"], "Projected 1")
            self.assertEqual(results[1]["bundle_status"], "EMPTY")
            self.assertEqual(results[1]["items"], [])
            self.assertIn("line 2", results[1]["error"])
            self.assertEqual(results[2]["items"][0]["projected_insight"], "Projected 2")
            self.assertEqual(llm.call_count, 2)

    def test_empty_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            input_path.write_text(
                "\n" + request_line("Goal", "Insight") + "\n\n",
                encoding="utf-8",
            )

            run_file(input_path, output_path, ProjectorService(SequencedFakeLLM()))

            self.assertEqual(
                len(output_path.read_text(encoding="utf-8").splitlines()),
                1,
            )

    def test_same_input_and_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(request_line("Goal", "Insight"), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_file(path, path, ProjectorService(SequencedFakeLLM()))


if __name__ == "__main__":
    unittest.main()
