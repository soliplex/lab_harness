import pytest

from soliplex_lab_harness import scoring
from soliplex_lab_harness.collect import ToolCall
from soliplex_lab_harness.records import TrialRecord

KNOWN = {"list_environments", "run", "run_python", "load_capability"}


def make_record(trial=0, *, calls=(), response="", ok=True, secs=1.0):
    return TrialRecord(
        cell="cell",
        trial=trial,
        elapsed_s=secs,
        ok=ok,
        response=response,
        tool_calls=[
            ToolCall(name=n, arguments=a) for n, a in calls
        ],
    )


def test_tally_counts_n_and_means():
    trials = [
        make_record(0, calls=[("run", "{}")], secs=2.0),
        make_record(1, calls=[("run", "{}"), ("run_python", "{}")], secs=4.0),
    ]

    result = scoring.tally("cell", trials, [scoring.succeeded()])

    assert result.n == 2
    assert result.mean_turns == 1.5
    assert result.mean_secs == 3.0


def test_rate_is_none_for_an_empty_cell():
    result = scoring.tally("cell", [], [scoring.succeeded()])

    assert result.rate("ok") is None


def test_response_contains_ignores_thousands_separators():
    check = scoring.response_contains("40935.89")

    assert check.test(make_record(response="Total: $40,935.89."))


def test_response_contains_can_require_exact_text():
    check = scoring.response_contains("40935.89", ignore_commas=False)

    assert not check.test(make_record(response="Total: $40,935.89."))


def test_invented_tool_name_flags_an_unknown_call():
    check = scoring.invented_tool_name(KNOWN)

    assert check.test(
        make_record(calls=[("bwrap-sandbox:run_python", "{}")])
    )


def test_invented_tool_name_passes_a_clean_run():
    check = scoring.invented_tool_name(KNOWN)

    assert not check.test(
        make_record(calls=[("run", "{}"), ("run_python", "{}")])
    )


def test_called_tool():
    check = scoring.called_tool("load_capability")

    assert check.test(make_record(calls=[("load_capability", "{}")]))


def test_invalid_argument_flags_a_guessed_value():
    check = scoring.invalid_argument(
        "environment_name", {"bare", "pandas-only", None}
    )

    assert check.test(
        make_record(calls=[("run", '{"environment_name": "???"}')])
    )


def test_invalid_argument_ignores_absent_argument():
    check = scoring.invalid_argument("environment_name", {"bare"})

    assert not check.test(make_record(calls=[("run", '{"command": "ls"}')]))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bwrap-sandbox:run_python", "namespaced"),
        ('run_python(environment_name="x")', "garbled-call"),
        ("run<tool_call|>", "garbled-call"),
        (None, "missing"),
        ("totally_made_up", "other"),
    ],
)
def test_classify_name(name, expected):
    assert scoring.classify_name(name) == expected


def test_bad_name_shapes_counts_by_shape():
    trials = [
        make_record(0, calls=[("a:b", "{}"), ("run", "{}")]),
        make_record(1, calls=[("run_python(x=1)", "{}")]),
    ]

    shapes = scoring.bad_name_shapes(trials, KNOWN)

    assert shapes == {"namespaced": 1, "garbled-call": 1}


def test_render_always_shows_n():
    checks = [scoring.succeeded()]
    tallies = [scoring.tally("before", [make_record()], checks)]

    table = scoring.render(tallies, checks)

    assert "cell" in table
    assert "n" in table
    assert "before" in table
    assert "100%" in table
