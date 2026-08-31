import pytest

from soliplex_lab_harness import preconditions


def make_result(**overrides):
    fields = {
        "cell": "v078-gemma4",
        "what": "deferral engages (expected: True)",
        "ok": True,
        "detail": "",
    }
    fields.update(overrides)
    return preconditions.Result(**fields)


def test_a_passing_result_renders_with_its_marker():
    result = make_result(what="a turn completes")

    rendered = str(result)

    assert rendered == "  ok   v078-gemma4: a turn completes"


def test_a_failing_result_renders_its_detail():
    result = make_result(ok=False, detail="load_capability seen: False")

    rendered = str(result)

    assert rendered.startswith("  FAIL v078-gemma4: ")
    assert rendered.endswith(" -- load_capability seen: False")


def test_a_result_without_detail_renders_no_trailing_dashes():
    result = make_result(ok=False, what="a turn completes")

    rendered = str(result)

    assert rendered == "  FAIL v078-gemma4: a turn completes"


def test_render_lists_every_result_then_the_verdict():
    results = [make_result(cell="a"), make_result(cell="b")]

    rendered = preconditions.render(results)

    assert rendered.splitlines()[:2] == [str(results[0]), str(results[1])]
    assert rendered.endswith("all 2 preconditions hold")


def test_render_counts_only_the_failures_in_its_verdict():
    results = [make_result(cell="a"), make_result(cell="b", ok=False)]

    rendered = preconditions.render(results)

    assert rendered.endswith("1 precondition(s) failed")


def test_render_says_when_nothing_was_checked():
    """Nothing checked is not the same as nothing wrong."""
    rendered = preconditions.render([])

    assert rendered == "no preconditions were checked"


def test_assert_ok_is_silent_when_every_check_holds():
    results = [make_result(cell="a"), make_result(cell="b")]

    preconditions.assert_ok(results)


def test_assert_ok_raises_naming_each_failure():
    results = [
        make_result(cell="a"),
        make_result(cell="b", ok=False, detail="no results"),
        make_result(cell="c", ok=False),
    ]

    with pytest.raises(preconditions.Failed) as caught:
        preconditions.assert_ok(results)

    assert caught.value.failures == [
        "b: deferral engages (expected: True) -- no results",
        "c: deferral engages (expected: True)",
    ]
    assert "2 precondition(s) failed" in str(caught.value)


def test_assert_ok_accepts_an_empty_list():
    """Nothing failed, so nothing is raised."""
    preconditions.assert_ok([])
