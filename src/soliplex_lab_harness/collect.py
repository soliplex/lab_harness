"""Capture the tool calls a run actually made, and what came back.

Why this reads a span *attribute* rather than tool spans: an invented tool
name never produces a tool span. ``ToolManager._resolve_tool`` rejects an
unknown name before ``wrap_tool_execute`` opens one, so harvesting spans
named ``running tool`` returns a clean sheet no matter what the model did.

The agent run span does carry the whole message history, as
``pydantic_ai.all_messages`` (set by
``Instrumentation._run_span_end_attributes``), and every tool-call part in it
is recorded under the name the model actually used. That attribute is also
what Logfire renders as a run's "events" list.

Responses are kept alongside the calls because a retry and a success are
indistinguishable without them: ``ToolReturnPart`` and ``RetryPromptPart``
both serialize as ``type='tool_call_response'`` carrying ``id``, ``name``
and ``result``, with no discriminator between them. Whether a call failed
is therefore a question about its ``result`` text -- see
``scoring.classify_result`` -- and can only be asked if the text is kept.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

ALL_MESSAGES_ATTR = "pydantic_ai.all_messages"


def _as_text(raw: Any) -> str:
    """A response payload as text, whatever shape it arrived in."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return str(raw)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call, under the name the model used for it."""

    name: str | None
    arguments: Any = None
    id: str | None = None

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


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What came back for one tool call, under the tool's own name.

    ``result`` is populated only when instrumentation was configured with
    ``include_content=True``, which ``drive.install_collector`` does.
    Without it the payload is absent and no outcome can be read.
    """

    name: str | None
    id: str | None = None
    result: Any = None

    @property
    def text(self) -> str:
        return _as_text(self.result)


def _parts(history: str | list[Any]) -> Iterator[dict[str, Any]]:
    """Every part of every message, in order."""
    messages = json.loads(history) if isinstance(history, str) else history
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or ():
            if isinstance(part, dict):
                yield part


def tool_calls_from_history(history: str | list[Any]) -> list[ToolCall]:
    """Extract tool calls, in order, from an OTel message history."""
    return [
        ToolCall(
            name=part.get("name"),
            arguments=part.get("arguments"),
            id=part.get("id"),
        )
        for part in _parts(history)
        if part.get("type") == "tool_call"
    ]


def tool_results_from_history(
    history: str | list[Any],
) -> list[ToolResult]:
    """Extract tool responses, in order, from an OTel message history."""
    return [
        ToolResult(
            name=part.get("name"),
            id=part.get("id"),
            result=part.get("result"),
        )
        for part in _parts(history)
        if part.get("type") == "tool_call_response"
    ]


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
    def take(
        self,
    ) -> tuple[list[ToolCall], list[ToolResult], str | None]:
        """Return ``(calls, results, raw_history)``, and reset.

        A single turn can end more than one agent run span -- a title agent
        alongside the room agent, for instance -- so the longest history is
        taken as the room turn. Returns ``([], [], None)`` when nothing was
        captured.
        """
        if not self.histories:
            return [], [], None
        raw = max(self.histories, key=len)
        self.histories.clear()
        return (
            tool_calls_from_history(raw),
            tool_results_from_history(raw),
            raw,
        )

    def reset(self) -> None:
        self.histories.clear()


def names(calls: Iterable[ToolCall]) -> list[str | None]:
    """The call names, in order -- handy for a one-line trial summary."""
    return [call.name for call in calls]
