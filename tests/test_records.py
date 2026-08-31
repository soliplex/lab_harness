import pytest

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


class Interrupted(Exception):
    """Stands in for a Ctrl-C partway through a run."""


def runner(seen, *, fail_at=None):
    """A ``run`` callable recording the trial indices it was handed."""

    def run(trial):
        seen.append(trial)
        if trial == fail_at:
            raise Interrupted
        return make_record(trial=trial)

    return run


def test_completed_counts_the_records_on_disk(tmp_path):
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(trial=0))
    records.append(path, make_record(trial=1))

    done = records.completed(path)

    assert done == 2


def test_completed_is_zero_when_the_file_does_not_exist(tmp_path):
    path = tmp_path / "absent.jsonl"

    done = records.completed(path)

    assert done == 0


def test_top_up_runs_the_whole_target_when_nothing_is_recorded(tmp_path):
    path = tmp_path / "cell.jsonl"
    seen = []

    ran = records.top_up(path, 3, runner(seen))

    assert ran == 3
    assert seen == [0, 1, 2]
    assert [r.trial for r in records.read(path)] == [0, 1, 2]


def test_top_up_counts_existing_records_toward_the_target(tmp_path):
    """``trials`` is a target, not a count: a smoke trial counts."""
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(trial=0))
    seen = []

    ran = records.top_up(path, 3, runner(seen))

    assert ran == 2
    assert seen == [1, 2]
    assert records.completed(path) == 3


def test_top_up_does_nothing_when_the_target_is_already_met(tmp_path):
    """Already satisfied is success, not an error."""
    path = tmp_path / "cell.jsonl"
    records.append(path, make_record(trial=0))
    seen = []

    ran = records.top_up(path, 1, runner(seen))

    assert ran == 0
    assert seen == []
    assert records.completed(path) == 1


def test_top_up_keeps_the_records_written_before_an_interrupt(tmp_path):
    """Appending as each arrives means an interrupt loses at most one."""
    path = tmp_path / "cell.jsonl"
    seen = []

    with pytest.raises(Interrupted):
        records.top_up(path, 5, runner(seen, fail_at=2))

    assert seen == [0, 1, 2]
    assert [r.trial for r in records.read(path)] == [0, 1]


def test_top_up_notifies_only_once_the_record_is_on_disk(tmp_path):
    path = tmp_path / "cell.jsonl"
    counts = []

    records.top_up(
        path,
        2,
        runner([]),
        on_trial=lambda record: counts.append(records.completed(path)),
    )

    assert counts == [1, 2]
