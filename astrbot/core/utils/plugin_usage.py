"""Execution context for attributing model token usage to plugins."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginUsageContext:
    """Identify the plugin and session that initiated a model generation.

    Args:
        plugin_id: Registered plugin identifier.
        umo: Unified message origin of the initiating plugin handler.
    """

    plugin_id: str
    umo: str


plugin_usage_context: ContextVar[PluginUsageContext | None] = ContextVar(
    "plugin_usage_context", default=None
)
