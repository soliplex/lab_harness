"""Capture the tool calls a run actually made.

Why this reads a span *attribute* rather than tool spans: an invented tool
name never produces a tool span. ``ToolManager._resolve_tool`` rejects an
unknown name before ``wrap_tool_execute`` opens one, so harvesting spans
named ``running tool`` returns a clean sheet no matter what the model did.

The agent run span does carry the whole message history, as
``pydantic_ai.all_messages`` (set by
``Instrumentation._run_span_end_attributes``), and every tool-call part in it
is recorded under the name the model actually used. That attribute is also
what Logfire renders as a run's "events" list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

ALL_MESSAGES_ATTR = "pydantic_ai.all_messages"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call, under the name the model used for it."""

    name: str | None
    arguments: Any = None

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Arguments as a mapping, or ``{}`` when they are not one.

        Instrumentation may hand back arguments as a JSON string or as an
        already-decoded object, so accept both and never raise.
        """
        raw = self.arguments
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return {}
        return raw if isinstance(raw, dict) else {}


def tool_calls_from_history(history: str | list[Any]) -> list[ToolCall]:
    """Extract tool calls, in order, from an OTel message history."""
    messages = json.loads(history) if isinstance(history, str) else history
    calls: list[ToolCall] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or ():
            if isinstance(part, dict) and part.get("type") == "tool_call":
                calls.append(
                    ToolCall(
                        name=part.get("name"),
                        arguments=part.get("arguments"),
                    )
                )
    return calls


class RunCollector:
    """An OpenTelemetry ``SpanProcessor`` that keeps agent-run histories.

    Duck-typed on purpose: nothing here imports opentelemetry, so the
    harness stays installable without it.
    """

    def __init__(self) -> None:
        self.histories: list[str] = []

    # -- SpanProcessor interface -------------------------------------
    def on_start(self, span: Any, parent_context: Any = None) -> None:
        """Required by the protocol; nothing to do at span start."""

    def on_end(self, span: Any) -> None:
        raw = (getattr(span, "attributes", None) or {}).get(
            ALL_MESSAGES_ATTR
        )
        if raw is not None:
            self.histories.append(raw)

    def shutdown(self) -> None:
        """Required by the protocol; the collector owns no resources."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Required by the protocol; nothing is buffered."""
        return True

    # -- consumption -------------------------------------------------
    def take(self) -> tuple[list[ToolCall], str | None]:
        """Return ``(calls, raw_history)`` for the run, and reset.

        A single turn can end more than one agent run span -- a title agent
        alongside the room agent, for instance -- so the longest history is
        taken as the room turn. Returns ``([], None)`` when nothing was
        captured.
        """
        if not self.histories:
            return [], None
        raw = max(self.histories, key=len)
        self.histories.clear()
        return tool_calls_from_history(raw), raw

    def reset(self) -> None:
        self.histories.clear()


def names(calls: Iterable[ToolCall]) -> list[str | None]:
    """The call names, in order -- handy for a one-line trial summary."""
    return [call.name for call in calls]
