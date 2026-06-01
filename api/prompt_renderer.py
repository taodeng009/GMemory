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
