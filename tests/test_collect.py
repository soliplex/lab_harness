import json

import pytest

from soliplex_lab_harness import collect


def make_history(*names_and_args):
    parts = [
        {"type": "tool_call", "name": name, "arguments": args}
        for name, args in names_and_args
    ]
    return json.dumps(
        [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]},
            {"role": "assistant", "parts": parts},
        ]
    )


def test_tool_calls_from_history_keeps_order():
    history = make_history(
        ("list_environments", "{}"),
        ("run_python", '{"environment_name": "bare"}'),
    )

    calls = collect.tool_calls_from_history(history)

    assert collect.names(calls) == ["list_environments", "run_python"]


def test_tool_calls_from_history_accepts_decoded_history():
    history = json.loads(make_history(("run", "{}")))

    calls = collect.tool_calls_from_history(history)

    assert collect.names(calls) == ["run"]


def test_tool_calls_from_history_ignores_non_tool_parts():
    history = json.dumps(
        [{"role": "assistant", "parts": [{"type": "text", "x": "y"}]}]
    )

    calls = collect.tool_calls_from_history(history)

    assert calls == []


def test_tool_calls_from_history_keeps_an_invented_name():
    """The name is recorded as the model wrote it, not as resolved."""
    history = make_history(("bwrap-sandbox:run_python", "{}"))

    calls = collect.tool_calls_from_history(history)

    assert collect.names(calls) == ["bwrap-sandbox:run_python"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"volume": "room"}', {"volume": "room"}),
        ({"volume": "room"}, {"volume": "room"}),
        ("not json at all", {}),
        ("[1, 2]", {}),
        (None, {}),
    ],
)
def test_parsed_arguments(raw, expected):
    call = collect.ToolCall(name="x", arguments=raw)

    assert call.parsed_arguments == expected


class FakeSpan:
    def __init__(self, attributes):
        self.attributes = attributes


def test_collector_take_returns_calls_and_resets():
    collector = collect.RunCollector()
    collector.on_end(
        FakeSpan({collect.ALL_MESSAGES_ATTR: make_history(("run", "{}"))})
    )

    calls, raw = collector.take()

    assert collect.names(calls) == ["run"]
    assert raw is not None
    assert collector.histories == []


def test_collector_take_prefers_the_longest_history():
    """A turn can end several run spans; the room turn is the longest."""
    collector = collect.RunCollector()
    short = make_history(("run", "{}"))
    long = make_history(("run", "{}"), ("run_python", "{}"))
    collector.on_end(FakeSpan({collect.ALL_MESSAGES_ATTR: short}))
    collector.on_end(FakeSpan({collect.ALL_MESSAGES_ATTR: long}))

    calls, _raw = collector.take()

    assert collect.names(calls) == ["run", "run_python"]


def test_collector_ignores_spans_without_the_attribute():
    collector = collect.RunCollector()
    collector.on_end(FakeSpan({"other": "value"}))
    collector.on_end(FakeSpan(None))

    calls, raw = collector.take()

    assert calls == []
    assert raw is None
