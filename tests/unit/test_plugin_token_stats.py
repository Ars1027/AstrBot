import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text
from sqlmodel import select

from astrbot.core.db.po import ProviderStat
from astrbot.core.pipeline.process_stage.stage import StarRequestSubStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.utils.plugin_usage import plugin_usage_context


@pytest.fixture
def plugin_runtime(temp_db, monkeypatch):
    """Connect real plugin dispatch and SDK methods to a deterministic provider."""
    provider = MagicMock(spec=Provider)
    provider.provider_config = {"id": "provider-1", "max_context_tokens": 0}
    provider.provider_settings = {}
    provider.get_model.return_value = "model-1"
    provider.meta.return_value = SimpleNamespace(id="provider-1")
    provider.text_chat = AsyncMock(
        return_value=LLMResponse(
            role="assistant",
            completion_text="done",
            usage=TokenUsage(input_other=11, input_cached=3, output=7),
        )
    )
    context = Context.__new__(Context)
    context._db = temp_db
    context.provider_manager = SimpleNamespace(
        get_provider_by_id=AsyncMock(return_value=provider),
    )
    events = []

    def event_for(handler, name="plugin-a"):
        """Register a handler and build an event accepted by the real SDK."""
        module = f"test_plugins.{name}"
        metadata = StarMetadata(name=name, author="author", module_path=module)
        monkeypatch.setitem(star_map, module, metadata)
        handlers = [
            SimpleNamespace(
                handler=handler,
                handler_module_path=module,
                handler_full_name=f"{module}.handle",
                handler_name="handle",
            )
        ]
        extras = {"activated_handlers": handlers, "handlers_parsed_params": {}}
        event = MagicMock(spec=AstrMessageEvent)
        event.unified_msg_origin = f"webchat:FriendMessage:{name}"
        event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
        event.is_stopped.return_value = False
        event.is_at_or_wake_command = False
        event.plugins_name = []
        events.append(event)
        return event

    yield SimpleNamespace(context=context, provider=provider, event_for=event_for)
    # Dispatch catches handler exceptions, including failed assertions in handlers.
    for event in events:
        event.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_generator_attribution_and_early_close(plugin_runtime):
    """Yielded requests keep identity while generator cleanup restores scope."""
    observed = []

    async def handler(event):
        try:
            for _ in range(2):
                observed.append(plugin_usage_context.get().plugin_id)
                yield ProviderRequest(prompt="hello")
        finally:
            observed.append(plugin_usage_context.get().plugin_id)

    stream = StarRequestSubStage().process(plugin_runtime.event_for(handler))
    for _ in range(2):
        request = await anext(stream)
        assert request.plugin_id == "author/plugin-a"
        assert plugin_usage_context.get() is None
    await stream.aclose()
    assert observed == ["author/plugin-a"] * 3
    assert plugin_usage_context.get() is None


@pytest.mark.asyncio
async def test_handler_cancellation_restores_context(plugin_runtime):
    """Cancellation closes the actual handler and does not leave its identity set."""
    entered = asyncio.Event()
    closed = []

    async def handler(event):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield None
        finally:
            closed.append(plugin_usage_context.get().plugin_id)

    async def dispatch():
        try:
            async for _ in StarRequestSubStage().process(
                plugin_runtime.event_for(handler)
            ):
                pass
        finally:
            assert plugin_usage_context.get() is None

    task = asyncio.create_task(dispatch())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == ["author/plugin-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["complete", "cancel", "error", "close"])
async def test_delegated_agent_records_once_on_completion_or_interruption(
    temp_db,
    plugin_runtime,
    monkeypatch,
    main_agent_build_config,
    interruption,
):
    """Dispatch attribution reaches the internal stage, including abnormal exits."""
    from astrbot.core.agent.response import AgentStats
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

    async def handler(event):
        yield ProviderRequest(prompt="test")

    event = plugin_runtime.event_for(handler)
    dispatch = StarRequestSubStage().process(event)
    request = await anext(dispatch)
    await dispatch.aclose()
    assert plugin_usage_context.get() is None
    runner = SimpleNamespace(
        provider=plugin_runtime.provider,
        stats=AgentStats(
            token_usage=TokenUsage(input_other=11, input_cached=3, output=7)
        ),
        was_aborted=lambda: False,
        get_final_llm_resp=lambda: LLMResponse(
            role="assistant", completion_text="done"
        ),
        run_context=SimpleNamespace(messages=[]),
    )
    event.get_extra.side_effect = lambda key, default=None: (
        request if key == "provider_request" else default
    )
    event.message_str = "test"
    event.message_obj = SimpleNamespace(message=[])
    event.platform_meta = SimpleNamespace(support_streaming_message=False)
    event.trace = MagicMock()
    stage = internal.InternalAgentSubStage()
    stage.ctx = SimpleNamespace(
        plugin_manager=SimpleNamespace(context=plugin_runtime.context)
    )
    stage.main_agent_cfg = main_agent_build_config
    stage.streaming_response = False
    stage.unsupported_streaming_strategy = "turn_off"
    stage.max_step = 5
    stage.show_tool_use = stage.show_tool_call_result = stage.show_reasoning = False
    stage.buffer_intermediate_messages = False
    stage._save_to_history = AsyncMock()
    plugin_runtime.provider.meta.return_value.type = "test-provider"
    monkeypatch.setattr(internal, "db_helper", temp_db)
    monkeypatch.setattr(internal.Metric, "upload", AsyncMock())
    persisted = asyncio.Event()
    original_record = internal._record_internal_agent_stats

    async def record(*args, **kwargs):
        await original_record(*args, **kwargs)
        persisted.set()

    recorder = AsyncMock(side_effect=record)
    monkeypatch.setattr(internal, "_record_internal_agent_stats", recorder)
    monkeypatch.setattr(
        internal,
        "build_main_agent",
        AsyncMock(
            return_value=SimpleNamespace(
                agent_runner=runner,
                provider_request=request,
                provider=plugin_runtime.provider,
                reset_coro=None,
            )
        ),
    )
    monkeypatch.setattr(internal, "call_event_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(internal, "try_capture_follow_up", lambda event: None)
    monkeypatch.setattr(internal, "register_active_runner", MagicMock())
    cleanup = MagicMock()
    monkeypatch.setattr(internal, "unregister_active_runner", cleanup)
    monkeypatch.setattr(
        internal, "extract_persona_custom_error_message_from_event", lambda event: None
    )
    entered = asyncio.Event()

    async def run(*args, **kwargs):
        entered.set()
        if interruption == "error":
            raise RuntimeError("agent failure")
        if interruption == "cancel":
            await asyncio.Event().wait()
        yield None

    monkeypatch.setattr(internal, "run_agent", run)
    stream = stage.process(event, "")
    if interruption == "cancel":
        task = asyncio.create_task(anext(stream))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif interruption == "close":
        await anext(stream)
        await stream.aclose()
    else:
        async for _ in stream:
            pass
    await asyncio.wait_for(persisted.wait(), timeout=5)
    recorder.assert_awaited_once()
    async with temp_db.get_db() as session:
        records = (await session.execute(select(ProviderStat))).scalars().all()
    assert len(records) == 1
    assert records[0].plugin_id == "author/plugin-a"
    assert records[0].agent_type == "internal"
    assert (
        records[0].status
        == {
            "complete": "completed",
            "error": "error",
            "cancel": "aborted",
            "close": "aborted",
        }[interruption]
    )
    assert records[0].token_output == 7
    cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_plugin_stat_migration_preserves_legacy_rows(temp_db):
    """Upgrade a populated old table twice without losing existing usage."""
    async with temp_db.engine.begin() as conn:
        await conn.run_sync(ProviderStat.__table__.create)
        await conn.execute(text("ALTER TABLE provider_stats DROP COLUMN plugin_id"))
        await conn.execute(
            text("""
            INSERT INTO provider_stats
                (created_at, updated_at, agent_type, status, umo, provider_id,
                 token_input_other, token_input_cached, token_output,
                 start_time, end_time, time_to_first_token)
            VALUES
                (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'internal', 'completed',
                 'legacy-session', 'provider-1', 11, 3, 7, 0, 0, 0)
        """)
        )

    await temp_db.initialize()
    await temp_db.initialize()
    async with temp_db.get_db() as session:
        records = (await session.execute(select(ProviderStat))).scalars().all()
        indexes = (
            await session.execute(text("PRAGMA index_list(provider_stats)"))
        ).all()
    assert len(records) == 1
    assert records[0].plugin_id is None
    assert records[0].token_input_other == 11
    assert records[0].token_input_cached == 3
    assert records[0].token_output == 7
    assert sum(row[1] == "ix_provider_stats_plugin_created_at" for row in indexes) == 1


@pytest.mark.asyncio
async def test_plugin_stat_optional_attribution_keeps_old_calls_valid(temp_db):
    """Old writers remain unassigned while new writers retain plugin identity."""
    old = await temp_db.insert_provider_stat(umo="session", provider_id="provider-1")
    new = await temp_db.insert_provider_stat(
        umo="session",
        provider_id="provider-1",
        plugin_id="author/plugin-a",
        agent_type="plugin",
        stats={"token_usage": {"input_other": 11, "input_cached": 3, "output": 7}},
    )
    assert old.plugin_id is None
    assert old.agent_type == "internal"
    assert new.plugin_id == "author/plugin-a"
    assert (new.token_input_other, new.token_input_cached, new.token_output) == (
        11,
        3,
        7,
    )
