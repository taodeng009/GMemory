import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


class ApiTracer:
    def __init__(
        self,
        trace_dir: str = "./.logs/hiagent_gmemory_api",
        enabled: bool = True,
        full_payload: bool = True,
        max_artifact_chars: int = 20000,
    ):
        self.trace_dir = Path(trace_dir)
        self.artifact_dir = self.trace_dir / "artifacts"
        self.enabled = enabled
        self.full_payload = full_payload
        self.max_artifact_chars = max_artifact_chars

    def new_trace_id(self) -> str:
        return uuid4().hex

    def record(
        self,
        trace_id: str,
        endpoint: str,
        request: dict[str, Any],
        derived: dict[str, Any],
        response: dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = f"{trace_id}.{endpoint.strip('/').split('/')[-1]}.json"
        artifact_path = self.artifact_dir / artifact_name
        artifact = {
            "request": request if self.full_payload else self._truncate_obj(request),
            "derived": derived,
            "response": response if self.full_payload else self._truncate_obj(response),
            "error": error,
        }
        artifact_text = json.dumps(artifact, ensure_ascii=False, indent=2, default=str)
        if len(artifact_text) > self.max_artifact_chars:
            artifact_text = artifact_text[: self.max_artifact_chars] + "\n...<truncated>"
        artifact_path.write_text(artifact_text, encoding="utf-8")

        summary = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "artifact": os.fspath(artifact_path),
            "error": error,
        }
        with (self.trace_dir / "traces.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

    def _truncate_obj(self, obj: Any) -> Any:
        text = json.dumps(obj, ensure_ascii=False, default=str)
        if len(text) <= self.max_artifact_chars:
            return obj
        return {"truncated": True, "preview": text[: self.max_artifact_chars]}
