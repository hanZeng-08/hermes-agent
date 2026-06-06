"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import patch


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"


class ConflictProvider:
    name = "conflict"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "external clarify"},
            {"name": "honcho_search", "description": "search"},
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return f'{{"source":"conflict","tool":"{tool_name}"}}'

    def shutdown(self):
        pass


def test_conflicting_memory_tool_removed_from_routing_table():
    """Built-in tools must shadow memory-provider tools with the same name (#40466)."""
    provider = ConflictProvider()
    cfg = {"memory": {"provider": "conflict"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                {"type": "function", "function": {"name": "clarify"}}
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is not None
    assert "clarify" not in agent._memory_manager._tool_to_provider, (
        "conflicting tool should be removed from _tool_to_provider"
    )
    assert "honcho_search" in agent._memory_manager._tool_to_provider, (
        "non-conflicting tool should remain"
    )
    assert "clarify" in agent.valid_tool_names


def test_builtin_clarify_shadows_memory_provider_at_dispatch():
    """invoke_tool must route 'clarify' to the built-in handler, not the memory provider (#40466)."""
    provider = ConflictProvider()
    cfg = {"memory": {"provider": "conflict"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                {"type": "function", "function": {"name": "clarify"}}
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is not None
    assert not agent._memory_manager.has_tool("clarify")

    with (
        patch("tools.clarify_tool.clarify_tool", return_value='{"source":"builtin"}') as mock_builtin,
        patch.object(
            agent._memory_manager,
            "handle_tool_call",
            side_effect=AssertionError("memory fallback should not run for clarify"),
        ) as mock_memory,
    ):
        result = agent._invoke_tool("clarify", {"question": "hello"}, "task-1")

    mock_builtin.assert_called_once()
    mock_memory.assert_not_called()
    assert result == '{"source":"builtin"}'
