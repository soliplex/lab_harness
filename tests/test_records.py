from soliplex_lab_harness import records
from soliplex_lab_harness.collect import ToolCall
from soliplex_lab_harness.collect import ToolResult


def make_record(**overrides):
    fields = {
        "cell": "before-gemma4",
        "trial": 0,
        "elapsed_s": 4.5,
        "ok": True,
        "response": "Total: 40,935.89",
        "tool_calls": [ToolCall(name="run", arguments='{"a": 1}')],
        "metadata": {"sha": "deadbeef"},
    }
    fields.update(overrides)
    return records.TrialRecord(**fields)


def test_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "nested" / "cell.jsonl"
    original = make_record()

    records.append(path, original)

    loaded = records.read(path)
    assert len(loaded) == 1
    assert loaded[0] == original


def test_append_accumulates(tmp_path):
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(trial=0))

    records.append(path, make_record(trial=1))

    assert [r.trial for r in records.read(path)] == [0, 1]


def test_from_dict_ignores_unknown_keys():
    data = make_record().as_dict() | {"invented_field": "ignored"}

    loaded = records.TrialRecord.from_dict(data)

    assert loaded.cell == "before-gemma4"


def test_call_names():
    record = make_record(
        tool_calls=[ToolCall(name="a"), ToolCall(name=None)]
    )

    assert record.call_names == ["a", None]


def test_metadata_survives_the_round_trip(tmp_path):
    """An arm that cannot be identified later is not a result."""
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(metadata={"ref": "v0.78.1"}))

    loaded = records.read(path)

    assert loaded[0].metadata == {"ref": "v0.78.1"}


def test_tool_results_default_to_unrecorded():
    """``None`` means the record cannot say, which is not the same as none."""
    record = make_record()

    assert record.tool_results is None


def test_unrecorded_outcomes_survive_the_round_trip(tmp_path):
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(tool_results=None))

    loaded = records.read(path)

    assert loaded[0].tool_results is None


def test_recorded_but_empty_outcomes_survive_the_round_trip(tmp_path):
    """``[]`` is a real answer: outcomes were captured, there were none."""
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(tool_results=[]))

    loaded = records.read(path)

    assert loaded[0].tool_results == []


def test_tool_results_survive_the_round_trip(tmp_path):
    path = tmp_path / "cell.jsonl"
    result = ToolResult(name="run", id="c1", result='{"error": "boom"}')
    records.append(path, make_record(tool_results=[result]))

    loaded = records.read(path)

    assert loaded[0].tool_results == [result]


def test_from_dict_reads_a_record_written_before_outcomes():
    """An older line still loads, and reports that it cannot say."""
    data = make_record().as_dict()
    del data["tool_results"]

    loaded = records.TrialRecord.from_dict(data)

    assert loaded.tool_results is None
