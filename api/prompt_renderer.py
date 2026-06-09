import re

from mas.memory.common import MASMessage


TASK_SOLVE_WITH_INSIGHTS = """
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.
Please use these as references to guide your thinking and approach to the current task:

{few_shots}
---

## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.
Pay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:

{memory_few_shots}
---

## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

{insights}
---

## Your Turn: Take Action!
Use the above examples and insights as a foundation, and now work on the following task:
{task_description}
"""

TASK_CONTEXT = """
### Task description:
{task_description}

### Key steps:
{key_steps}

### Detailed trajectory:
{trajectory}
"""

KEY_STEPS_ONLY_MEMORY = """## Retrieved Long-Term Memory
BEGIN_RETRIEVED_MEMORY

This is past experience from similar successful tasks.
Use it only as a high-level strategy reference.
The current task, current observation, and valid actions always take priority.
Do not copy object names, receptacle names, locations, or numbers from past tasks.
Reuse only the general procedure when it matches the current situation.

### Past Successful Tasks

{tasks}

END_RETRIEVED_MEMORY
"""

KEY_STEPS_ONLY_TASK = """Task {idx}:
Past task description:
{task_description}

Useful key steps:
{key_steps}"""

GOAL_KEY_STEPS_ONLY_TASK = """Task {idx}:
Past task goal:
{task_goal}

Useful key steps:
{key_steps}"""

INSIGHT_ONLY_MEMORY = """## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

{insights}
---
"""

_BECAUSE_SUFFIX_RE = re.compile(r"\s*,?\s+because\b.*$", flags=re.IGNORECASE)
_BECAUSE_WORD_RE = re.compile(r"\bbecause\b", flags=re.IGNORECASE)


def render_memory_prompt(successful: list[MASMessage], insights: list[str], task_description: str) -> str:
    if not successful and not insights:
        return ""

    memory_few_shots = "\n\n".join(
        f"Task {idx + 1}:\n"
        + TASK_CONTEXT.format(
            task_description=item.task_description,
            key_steps=item.get_extra_field("key_steps"),
            trajectory=item.task_trajectory,
        )
        for idx, item in enumerate(successful)
    )
    insight_text = "\n".join(f"{idx}. {insight}" for idx, insight in enumerate(insights, 1))
    return TASK_SOLVE_WITH_INSIGHTS.format(
        few_shots="",
        memory_few_shots=memory_few_shots,
        insights=insight_text,
        task_description=task_description,
    )


def render_key_steps_only_memory_prompt(successful: list[MASMessage]) -> str:
    if not successful:
        return ""

    tasks = "\n\n".join(
        KEY_STEPS_ONLY_TASK.format(
            idx=idx + 1,
            task_description=item.task_description or "",
            key_steps=item.get_extra_field("key_steps") or "",
        )
        for idx, item in enumerate(successful)
    )
    return KEY_STEPS_ONLY_MEMORY.format(tasks=tasks)


def render_goal_key_steps_only_memory_prompt(successful: list[MASMessage]) -> str:
    if not successful:
        return ""

    tasks = "\n\n".join(
        GOAL_KEY_STEPS_ONLY_TASK.format(
            idx=idx + 1,
            task_goal=_extract_task_goal(item),
            key_steps=item.get_extra_field("key_steps") or "",
        )
        for idx, item in enumerate(successful)
    )
    return KEY_STEPS_ONLY_MEMORY.format(tasks=tasks)


def render_insight_only_memory_prompt(insights: list[str], insight_style: str = "original") -> str:
    if not insights:
        return ""

    insight_text = "\n".join(
        f"{idx}. {normalize_insight_text(insight, insight_style)}"
        for idx, insight in enumerate(insights, 1)
    )
    return INSIGHT_ONLY_MEMORY.format(insights=insight_text)


def normalize_insight_text(insight: str, insight_style: str = "original") -> str:
    normalized = (insight or "").strip()
    if insight_style == "no_because":
        return remove_because_clause(normalized)
    return normalized


def remove_because_clause(insight: str) -> str:
    shortened = _BECAUSE_SUFFIX_RE.sub("", insight.strip()).strip()
    shortened = shortened.rstrip(" ,;:")
    if shortened and shortened[-1] not in ".!?":
        shortened += "."
    return shortened


def count_because_lines(insights: list[str]) -> int:
    return sum(1 for insight in insights if _BECAUSE_WORD_RE.search(insight or ""))


def _extract_task_goal(item: MASMessage) -> str:
    task_main = (item.task_main or "").strip()
    if task_main.lower().startswith("alfworld-"):
        return task_main[len("alfworld-") :].strip()
    if task_main:
        return task_main

    task_description = item.task_description or ""
    match = re.search(r"\*\*Here is your task:\s*(?P<goal>.*?)(?:\n|$)", task_description, re.DOTALL)
    if match:
        return match.group("goal").strip()
    return task_description.strip()
