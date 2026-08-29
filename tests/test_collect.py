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

    calls, results, raw = collector.take()

    assert collect.names(calls) == ["run"]
    assert results == []
    assert raw is not None
    assert collector.histories == []


def test_collector_take_prefers_the_longest_history():
    """A turn can end several run spans; the room turn is the longest."""
    collector = collect.RunCollector()
    short = make_history(("run", "{}"))
    long = make_history(("run", "{}"), ("run_python", "{}"))
    collector.on_end(FakeSpan({collect.ALL_MESSAGES_ATTR: short}))
    collector.on_end(FakeSpan({collect.ALL_MESSAGES_ATTR: long}))

    calls, _results, _raw = collector.take()

    assert collect.names(calls) == ["run", "run_python"]


def test_collector_ignores_spans_without_the_attribute():
    collector = collect.RunCollector()
    collector.on_end(FakeSpan({"other": "value"}))
    collector.on_end(FakeSpan(None))

    calls, results, raw = collector.take()

    assert calls == []
    assert results == []
    assert raw is None


def make_response_history(*names_ids_results):
    parts = [
        {
            "type": "tool_call_response",
            "name": name,
            "id": call_id,
            "result": result,
        }
        for name, call_id, result in names_ids_results
    ]
    return json.dumps([{"role": "user", "parts": parts}])


def test_tool_results_from_history_keeps_order():
    history = make_response_history(
        ("list_environments", "a", "['bare']"),
        ("run_python", "b", "40935.89"),
    )

    results = collect.tool_results_from_history(history)

    assert [r.name for r in results] == ["list_environments", "run_python"]


def test_tool_results_from_history_ignores_calls():
    """A call part and a response part share a message, not a type."""
    history = make_history(("run", "{}"))

    results = collect.tool_results_from_history(history)

    assert results == []


def test_tool_results_carry_the_call_id():
    """The id is what ties a response back to the call it answers."""
    history = make_response_history(("run", "call_1", "ok"))

    results = collect.tool_results_from_history(history)

    assert results[0].id == "call_1"


def test_tool_calls_carry_the_call_id():
    history = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "name": "run", "id": "call_1"}
                ],
            }
        ]
    )

    calls = collect.tool_calls_from_history(history)

    assert calls[0].id == "call_1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain text", "plain text"),
        ({"error": "boom"}, '{"error": "boom"}'),
        ([1, 2], "[1, 2]"),
        (None, ""),
    ],
)
def test_tool_result_text(raw, expected):
    result = collect.ToolResult(name="run", result=raw)

    assert result.text == expected


def test_collector_take_returns_results():
    collector = collect.RunCollector()
    history = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "name": "run", "id": "c1"},
                    {
                        "type": "tool_call_response",
                        "name": "run",
                        "id": "c1",
                        "result": "ok",
                    },
                ],
            }
        ]
    )
    collector.on_end(FakeSpan({collect.ALL_MESSAGES_ATTR: history}))

    calls, results, _raw = collector.take()

    assert collect.names(calls) == ["run"]
    assert [r.text for r in results] == ["ok"]
