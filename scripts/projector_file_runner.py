import argparse
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from api.projector import ProjectorService
from api.schemas import ProjectorRequest, ProjectorResponse
from api.tracing import ApiTracer


DEFAULT_MODEL = "gpt-3.5-turbo-0125"


def run_file(
    input_path: str | Path,
    output_path: str | Path,
    projector_service: ProjectorService,
) -> None:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                request = ProjectorRequest.model_validate_json(line)
                response = projector_service.project(request)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                response = ProjectorResponse(
                    bundle_status="EMPTY",
                    items=[],
                    error=_summarize_line_error(line_number, exc),
                )
            destination.write(response.model_dump_json() + "\n")
            destination.flush()


def build_projector_service() -> ProjectorService:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("OPENAI_API_BASE", "")
    os.environ.setdefault("OPENAI_API_KEY", "")

    from mas.llm import GPTChat

    model_name = os.getenv("GMEMORY_API_MODEL", DEFAULT_MODEL)
    return ProjectorService(
        llm_client=GPTChat(model_name=model_name),
        tracer=ApiTracer(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project raw insights from a UTF-8 JSONL input file."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args(argv)

    run_file(args.input, args.output, build_projector_service())
    return 0


def _summarize_line_error(line_number: int, exc: Exception) -> str:
    return f"line {line_number}: {exc.__class__.__name__}: {str(exc)[:500]}"


if __name__ == "__main__":
    raise SystemExit(main())
