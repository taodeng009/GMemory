#!/usr/bin/env python3
"""Recompute similarity for GMemory API retrieve artifacts.

The script reads .logs/hiagent_gmemory_api/artifacts/*.retrieve.json,
extracts the query_task and the historical tasks rendered into memory_prompt,
then computes cosine similarity using the same EmbeddingFunc used by GMemory.

Run this on a server where the configured sentence-transformers model is
available locally or can be loaded by SentenceTransformer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


DEFAULT_ARTIFACT_DIR = Path(".logs/hiagent_gmemory_api/artifacts")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TASK_BLOCK_RE = re.compile(
    r"Task\s+(?P<index>\d+):\s*"
    r"### Task description:\s*"
    r"(?P<description>.*?)(?=\n### Key steps:|\nTask\s+\d+:|\Z)",
    re.DOTALL,
)
TASK_GOAL_RE = re.compile(
    r"\*\*Here is your task:\s*(?P<goal>.*?)(?=\n|$)",
    re.DOTALL,
)
GOAL_PREFIX_RE = re.compile(
    r"^\s*The goal is to satisfy the following conditions:\s*",
    re.IGNORECASE,
)
ALFWORLD_PREFIX_RE = re.compile(r"^\s*alfworld-", re.IGNORECASE)


@dataclass
class SimilarityRow:
    trace_id: str
    artifact: str
    memory_size: int
    successful_count: int
    failed_count: int
    insight_count: int
    returned_task_index: int
    similarity: float | None
    status: str
    query_task: str
    returned_task: str
    query_embedding_text: str
    returned_embedding_text: str
    error: str


class EmbeddingFunc:
    def __init__(self, model_type: str):
        self.model = SentenceTransformer(model_type)

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute embedding cosine similarity for GMemory retrieve artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Directory containing *.retrieve.json artifacts.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "SentenceTransformer model/path. Defaults to GMEMORY_API_EMBEDDING_MODEL "
            "from .env, then sentence-transformers/all-MiniLM-L6-v2."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include retrieve artifacts that returned no historical task.",
    )
    parser.add_argument(
        "--strip-goal-prefix",
        action="store_true",
        help=(
            "Remove the fixed PDDL prefix 'The goal is to satisfy the following "
            "conditions:' before embedding query and returned tasks."
        ),
    )
    parser.add_argument(
        "--strip-alfworld-prefix",
        action="store_true",
        help=(
            "Remove the fixed ALFWorld namespace prefix 'alfworld-' before "
            "embedding query and returned tasks."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first malformed artifact instead of reporting a row with status=error.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_embedding_text(
    text: str,
    strip_goal_prefix: bool,
    strip_alfworld_prefix: bool,
) -> str:
    text = normalize_space(text)
    if strip_alfworld_prefix:
        text = ALFWORLD_PREFIX_RE.sub("", text)
    if strip_goal_prefix:
        text = GOAL_PREFIX_RE.sub("", text)
    return normalize_space(text)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    left = np.array(vec1)
    right = np.array(vec2)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_task_for_comparison(text: str, query_task: str) -> str:
    text = normalize_space(text)
    if ALFWORLD_PREFIX_RE.match(query_task) and not ALFWORLD_PREFIX_RE.match(text):
        return f"alfworld-{text}"
    return text


def extract_returned_tasks(memory_prompt: str, query_task: str) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    for match in TASK_BLOCK_RE.finditer(memory_prompt or ""):
        index = int(match.group("index"))
        description = match.group("description").strip()
        goal_match = TASK_GOAL_RE.search(description)
        task = goal_match.group("goal") if goal_match else description
        tasks.append((index, normalize_task_for_comparison(task, query_task)))
    return tasks


def get_trace_id(path: Path, data: dict[str, Any]) -> str:
    response = data.get("response", {})
    return response.get("trace_id") or data.get("trace_id") or path.name.replace(".retrieve.json", "")


def get_stats(data: dict[str, Any]) -> dict[str, int]:
    stats = data.get("response", {}).get("stats", {}) or {}
    return {
        "memory_size": int(stats.get("memory_size", 0) or 0),
        "successful_count": int(stats.get("successful_count", 0) or 0),
        "failed_count": int(stats.get("failed_count", 0) or 0),
        "insight_count": int(stats.get("insight_count", 0) or 0),
    }


def empty_row(path: Path, data: dict[str, Any], status: str, error: str = "") -> SimilarityRow:
    stats = get_stats(data)
    return SimilarityRow(
        trace_id=get_trace_id(path, data),
        artifact=str(path),
        returned_task_index=0,
        similarity=None,
        status=status,
        query_task=normalize_space(data.get("derived", {}).get("query_task", "")),
        returned_task="",
        query_embedding_text="",
        returned_embedding_text="",
        error=error,
        **stats,
    )


def analyze_artifact(
    path: Path,
    embedder: EmbeddingFunc,
    include_empty: bool,
    strip_goal_prefix: bool,
    strip_alfworld_prefix: bool,
) -> list[SimilarityRow]:
    data = load_artifact(path)
    query_task = normalize_space(data.get("derived", {}).get("query_task", ""))
    memory_prompt = data.get("response", {}).get("memory_prompt", "")
    returned_tasks = extract_returned_tasks(memory_prompt, query_task)

    if not query_task:
        return [empty_row(path, data, "parse_failed", "missing derived.query_task")]

    if not returned_tasks:
        return [empty_row(path, data, "empty", "no returned task parsed")] if include_empty else []

    query_embedding_text = normalize_embedding_text(
        query_task,
        strip_goal_prefix,
        strip_alfworld_prefix,
    )
    query_embedding = embedder.embed_query(query_embedding_text)
    stats = get_stats(data)
    rows: list[SimilarityRow] = []
    for index, returned_task in returned_tasks:
        returned_embedding_text = normalize_embedding_text(
            returned_task,
            strip_goal_prefix,
            strip_alfworld_prefix,
        )
        returned_embedding = embedder.embed_query(returned_embedding_text)
        similarity = cosine_similarity(query_embedding, returned_embedding)
        rows.append(
            SimilarityRow(
                trace_id=get_trace_id(path, data),
                artifact=str(path),
                returned_task_index=index,
                similarity=similarity,
                status="ok",
                query_task=query_embedding_text,
                returned_task=returned_embedding_text,
                query_embedding_text=query_embedding_text,
                returned_embedding_text=returned_embedding_text,
                error="",
                **stats,
            )
        )
    return rows


def collect_rows(
    artifact_dir: Path,
    embedder: EmbeddingFunc,
    include_empty: bool,
    fail_fast: bool,
    strip_goal_prefix: bool,
    strip_alfworld_prefix: bool,
) -> list[SimilarityRow]:
    rows: list[SimilarityRow] = []
    paths = sorted(artifact_dir.glob("*.retrieve.json"), key=lambda item: item.stat().st_mtime)
    for path in paths:
        try:
            rows.extend(
                analyze_artifact(
                    path,
                    embedder,
                    include_empty,
                    strip_goal_prefix,
                    strip_alfworld_prefix,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            rows.append(
                SimilarityRow(
                    trace_id=path.name.replace(".retrieve.json", ""),
                    artifact=str(path),
                    memory_size=0,
                    successful_count=0,
                    failed_count=0,
                    insight_count=0,
                    returned_task_index=0,
                    similarity=None,
                    status="error",
                    query_task="",
                    returned_task="",
                    query_embedding_text="",
                    returned_embedding_text="",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            )
    return rows


def print_table(rows: list[SimilarityRow]) -> None:
    if not rows:
        print("No rows found.")
        return
    print("\t".join(["trace_id", "mem", "succ", "task", "similarity", "status", "error"]))
    for row in rows:
        similarity = "" if row.similarity is None else f"{row.similarity:.6f}"
        print(
            "\t".join(
                [
                    row.trace_id,
                    str(row.memory_size),
                    str(row.successful_count),
                    str(row.returned_task_index),
                    similarity,
                    row.status,
                    row.error,
                ]
            )
        )


def print_csv(rows: list[SimilarityRow]) -> None:
    fieldnames = list(SimilarityRow.__dataclass_fields__.keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    load_dotenv()

    model = (
        args.embedding_model
        or os.getenv("GMEMORY_API_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    embedder = EmbeddingFunc(model)
    rows = collect_rows(
        Path(args.artifact_dir),
        embedder,
        args.include_empty,
        args.fail_fast,
        args.strip_goal_prefix,
        args.strip_alfworld_prefix,
    )

    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False))
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
