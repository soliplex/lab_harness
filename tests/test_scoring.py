import pytest

from soliplex_lab_harness import scoring
from soliplex_lab_harness.collect import ToolCall
from soliplex_lab_harness.collect import ToolResult
from soliplex_lab_harness.records import TrialRecord

KNOWN = {"list_environments", "run", "run_python", "load_capability"}


def make_record(
    trial=0, *, calls=(), response="", ok=True, secs=1.0, results=None
):
    return TrialRecord(
        cell="cell",
        trial=trial,
        elapsed_s=secs,
        ok=ok,
        response=response,
        tool_calls=[
            ToolCall(name=n, arguments=a) for n, a in calls
        ],
        tool_results=results,
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


RETRY_UNKNOWN = (
    "Unknown tool name: 'bwrap-sandbox:run'. Available tools: run\n\n"
    "Fix the errors and try again."
)
RETRY_ALREADY = (
    "Capability 'reporting' is already available.\n\n"
    "Fix the errors and try again."
)
RETRY_VALIDATION = (
    "1 validation error:\n```json\n[]\n```\n\n"
    "Fix the errors and try again."
)
TOOL_ERROR = '[{"error": "mermaidx: not found"}]'
PLAIN_OK = "40935.89"


def result(text, name="run"):
    return ToolResult(name=name, id="c1", result=text)


# -- distributions --------------------------------------------------------


def test_summarize_reports_dispersion():
    values = [10.0, 11.0, 49.0]

    dist = scoring.summarize(values)

    assert dist.n == 3
    assert dist.median == 11.0
    assert dist.maximum == 49.0
    assert dist.sd == 22.2


def test_summarize_of_nothing_is_all_none():
    dist = scoring.summarize([])

    assert dist.n == 0
    assert dist.mean is None
    assert dist.maximum is None


def test_summarize_skips_unknown_values():
    """A trial that cannot answer contributes nothing, not a zero."""
    dist = scoring.summarize([4.0, None, 6.0])

    assert dist.n == 2
    assert dist.mean == 5.0


def test_summarize_sd_of_one_value_is_zero():
    dist = scoring.summarize([7.0])

    assert dist.sd == 0.0


def test_tally_summarizes_turns_and_secs():
    trials = [
        make_record(0, calls=[("run", "{}")], secs=2.0),
        make_record(1, calls=[("run", "{}"), ("run", "{}")], secs=4.0),
    ]

    tallied = scoring.tally("cell", trials, [scoring.succeeded()])

    assert tallied.turns.median == 1.5
    assert tallied.turns.maximum == 2.0
    assert tallied.secs.mean == 3.0


def test_secs_per_turn_separates_more_turns_from_slower_turns():
    trials = [make_record(0, calls=[("run", "{}")] * 4, secs=8.0)]

    tallied = scoring.tally("cell", trials, [])

    assert tallied.secs_per_turn == 2.0


def test_secs_per_turn_is_none_without_turns():
    trials = [make_record(0, calls=[], secs=8.0)]

    tallied = scoring.tally("cell", trials, [])

    assert tallied.secs_per_turn is None


def test_render_distributions_has_a_row_per_cell():
    tallies = [
        scoring.tally("a", [make_record(0, calls=[("run", "{}")])], []),
        scoring.tally("b", [make_record(0, calls=[("run", "{}")])], []),
    ]

    table = scoring.render_distributions(tallies, "turns")

    assert len(table.splitlines()) == 4


def test_render_distributions_shows_unknown_as_a_dash():
    """Retries of a record that predates outcomes must not read as zero."""
    tallies = [scoring.tally("a", [make_record(0)], [])]

    table = scoring.render_distributions(tallies, "retries")

    assert table.splitlines()[-1].split() == [
        "a",
        "0",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]


def test_render_distributions_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="field must be one of"):
        scoring.render_distributions([], "correctness")


# -- outcomes -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (RETRY_UNKNOWN, "retry-unknown-tool"),
        (RETRY_ALREADY, "retry-already-available"),
        (RETRY_VALIDATION, "retry-validation"),
        ("Something else.\n\nFix the errors and try again.", "retry-other"),
        (TOOL_ERROR, "tool-error"),
        ('{"error": "boom"}', "tool-error"),
        (PLAIN_OK, "ok"),
        ("", "ok"),
        ("[]", "ok"),
    ],
)
def test_classify_result(text, expected):
    assert scoring.classify_result(text) == expected


def test_retry_count_is_none_when_outcomes_were_not_recorded():
    """The distinction this whole change exists for."""
    record = make_record(results=None)

    assert scoring.retry_count(record) is None


def test_retry_count_is_zero_when_recorded_and_clean():
    record = make_record(results=[result(PLAIN_OK)])

    assert scoring.retry_count(record) == 0


def test_retry_count_counts_retries_only():
    record = make_record(
        results=[result(RETRY_UNKNOWN), result(PLAIN_OK), result(TOOL_ERROR)]
    )

    assert scoring.retry_count(record) == 1


def test_retried_abstains_without_outcomes():
    check = scoring.retried()

    assert check.test(make_record(results=None)) is None


def test_retried_flags_a_retry():
    check = scoring.retried()

    assert check.test(make_record(results=[result(RETRY_UNKNOWN)]))


def test_tool_errored_flags_a_tool_failure():
    check = scoring.tool_errored()

    assert check.test(make_record(results=[result(TOOL_ERROR)]))


def test_rate_excludes_trials_that_cannot_answer():
    """Two trials, one answerable and retried: the rate is 100%, not 50%."""
    trials = [
        make_record(0, results=None),
        make_record(1, results=[result(RETRY_UNKNOWN)]),
    ]

    tallied = scoring.tally("cell", trials, [scoring.retried()])

    assert tallied.n == 2
    assert tallied.unknowns["retry"] == 1
    assert tallied.rate("retry") == 1.0


def test_rate_is_none_when_no_trial_can_answer():
    trials = [make_record(0, results=None), make_record(1, results=None)]

    tallied = scoring.tally("cell", trials, [scoring.retried()])

    assert tallied.rate("retry") is None


def test_call_count():
    record = make_record(
        calls=[("run", "{}"), ("run", "{}"), ("run_python", "{}")]
    )

    assert scoring.call_count(record, "run") == 2


def test_called_repeatedly_ignores_a_single_call():
    check = scoring.called_repeatedly("list_environments")

    assert not check.test(
        make_record(calls=[("list_environments", "{}")])
    )


def test_called_repeatedly_flags_a_second_call():
    check = scoring.called_repeatedly("list_environments")

    assert check.test(
        make_record(
            calls=[("list_environments", "{}"), ("list_environments", "{}")]
        )
    )


def test_result_shapes_counts_across_records():
    trials = [
        make_record(0, results=[result(RETRY_UNKNOWN), result(PLAIN_OK)]),
        make_record(1, results=[result(TOOL_ERROR)]),
        make_record(2, results=None),
    ]

    shapes = scoring.result_shapes(trials)

    assert shapes == {"retry-unknown-tool": 1, "tool-error": 1}
