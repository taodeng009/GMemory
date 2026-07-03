from api.service import GMemoryApiConfig, GMemoryApiService


def test_merge_config_defaults(monkeypatch):
    monkeypatch.delenv("GMEMORY_API_MERGE", raising=False)
    monkeypatch.delenv("GMEMORY_API_MERGE_STEPS", raising=False)

    service = GMemoryApiService(config=GMemoryApiConfig())

    assert service.config.merge_enabled is True
    assert service.config.merge_steps == 20


def test_merge_config_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GMEMORY_API_MERGE", "disabled")
    monkeypatch.setenv("GMEMORY_API_MERGE_STEPS", "10")

    service = GMemoryApiService(config=GMemoryApiConfig())

    assert service.config.merge_enabled is False
    assert service.config.merge_steps == 10


def test_invalid_merge_config_uses_safe_defaults(monkeypatch):
    monkeypatch.setenv("GMEMORY_API_MERGE", "unexpected")
    monkeypatch.setenv("GMEMORY_API_MERGE_STEPS", "0")

    service = GMemoryApiService(config=GMemoryApiConfig())

    assert service.config.merge_enabled is True
    assert service.config.merge_steps == 20


def test_non_numeric_merge_steps_uses_configured_default(monkeypatch):
    monkeypatch.setenv("GMEMORY_API_MERGE_STEPS", "not-a-number")

    service = GMemoryApiService(config=GMemoryApiConfig(merge_steps=12))

    assert service.config.merge_steps == 12
