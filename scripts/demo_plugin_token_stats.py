"""Verify plugin token collection without calling an external model.

Run from the repository root with:
    uv run python scripts/demo_plugin_token_stats.py
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.pipeline.process_stage.stage import StarRequestSubStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.dashboard.services.stat_service import StatService


async def main() -> None:
    """Exercise real plugin dispatch, SDK collection, persistence, and aggregation.

    Raises:
        AssertionError: Collected token usage differs from the fixed provider data.
    """
    temp_root = Path(get_astrbot_temp_path())
    temp_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="plugin-token-demo-", dir=temp_root) as directory:
        db = SQLiteDatabase(str(Path(directory) / "demo.db"))
        plugins = [
            StarMetadata(name="alpha", author="demo", display_name="Alpha"),
            StarMetadata(name="beta", author="demo", display_name="Beta"),
        ]
        provider = MagicMock(spec=Provider)
        provider.get_model.return_value = "demo-model"
        context = Context.__new__(Context)
        context._db = db
        context.provider_manager = SimpleNamespace(
            get_provider_by_id=AsyncMock(return_value=provider)
        )
        try:
            for multiplier, plugin in enumerate(plugins, start=1):
                provider.text_chat = AsyncMock(
                    return_value=LLMResponse(
                        role="assistant",
                        completion_text="Demo response",
                        usage=TokenUsage(
                            input_other=11 * multiplier,
                            input_cached=3 * multiplier,
                            output=7 * multiplier,
                        ),
                    )
                )

                async def handler(event):
                    """Generate a response through the real plugin SDK."""
                    await context.llm_generate(
                        chat_provider_id="demo-provider", prompt="Hello"
                    )

                module = f"__plugin_token_demo__.{plugin.name}"
                star_map[module] = plugin
                metadata = SimpleNamespace(
                    handler=handler,
                    handler_module_path=module,
                    handler_full_name=f"{module}.handle",
                    handler_name="handle",
                )
                extras = {
                    "activated_handlers": [metadata],
                    "handlers_parsed_params": {},
                }
                event = MagicMock(spec=AstrMessageEvent)
                event.unified_msg_origin = f"webchat:FriendMessage:{plugin.name}"
                event.get_extra.side_effect = lambda key, default=None: extras.get(
                    key, default
                )
                event.is_stopped.return_value = False
                event.is_at_or_wake_command = False
                event.plugins_name = []
                async for _ in StarRequestSubStage().process(event):
                    pass

            service = StatService(
                db,
                SimpleNamespace(
                    star_context=SimpleNamespace(
                        get_all_stars=lambda: plugins,
                    )
                ),
                {},
            )
            result = await service.get_plugin_token_stats(1)
            totals = {row["plugin_id"]: row["total_tokens"] for row in result["items"]}
            assert totals == {"demo/alpha": 21, "demo/beta": 42}, totals
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("Demo passed: Alpha = 21 tokens; Beta = 42 tokens.")
        finally:
            await db.engine.dispose()
            for plugin in plugins:
                star_map.pop(f"__plugin_token_demo__.{plugin.name}", None)


if __name__ == "__main__":
    asyncio.run(main())
