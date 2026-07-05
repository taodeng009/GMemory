from unittest.mock import Mock

from mas.memory.mas_memory.GMemory import InsightsManager


def make_manager(responses):
    manager = object.__new__(InsightsManager)
    manager.merge_strategy = "atomic_v1"
    manager.atomic_merge_ratio = 0.33
    manager.atomic_merge_max_words = 30
    manager.logger = Mock()
    manager.llm_model = Mock(side_effect=responses)
    return manager


def numbered_rules(count):
    return "\n".join(f"{index}. Keep strategy {index} separate." for index in range(1, count + 1))


def test_atomic_merge_uses_ceiling_ratio_limit():
    manager = make_manager([numbered_rules(4)])
    source_rules = [f"Source rule {index}." for index in range(10)]

    result = manager._merge_rules_atomic_v1(source_rules)

    assert len(result) == 4
    messages = manager.llm_model.call_args.args[0]
    assert "no more than 4 atomic insights" in messages[1].content


def test_atomic_merge_retries_once_after_basic_validation_failure():
    too_long = " ".join(["word"] * 31)
    manager = make_manager([f"1. {too_long}", "1. Keep the rule concise."])

    result = manager._merge_rules_atomic_v1(["Source rule."])

    assert result == ["Keep the rule concise."]
    assert manager.llm_model.call_count == 2
    retry_messages = manager.llm_model.call_args.args[0]
    assert "Validation feedback from the previous output" in retry_messages[1].content


def test_atomic_merge_falls_back_to_source_rules_after_second_failure():
    manager = make_manager(["invalid output", "still invalid"])
    source_rules = ["First source rule.", "Second source rule."]

    result = manager._merge_rules_atomic_v1(source_rules)

    assert result == source_rules
    assert manager.llm_model.call_count == 2
    manager.logger.info.assert_any_call("Fallback to source rules: True")
