"""Turn trial records into rates.

Two habits are baked in, both learned the hard way:

* rates, not outcomes -- one run tells you nothing about a stochastic system
* report N alongside every rate, because at N=20 a one- or two-run
  difference is not a result
* ``None`` is not zero -- a record that cannot answer a question is
  excluded from that question's denominator rather than counted as a miss,
  because a null that reads as a result is this package's whole subject
"""

from __future__ import annotations

import collections
import dataclasses
import json
import statistics
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

from .records import TrialRecord

#: ``None`` means the trial cannot answer -- not that the answer is no.
Predicate = Callable[[TrialRecord], bool | None]


@dataclasses.dataclass(frozen=True, slots=True)
class Check:
    """A named per-trial yes/no question."""

    name: str
    test: Predicate


@dataclasses.dataclass(frozen=True, slots=True)
class Dist:
    """A distribution summary.

    ``n`` counts the values that were *knowable*: a metric a record cannot
    answer contributes nothing rather than a zero. Every field is ``None``
    when nothing was knowable.
    """

    n: int = 0
    mean: float | None = None
    sd: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None


def summarize(values: Iterable[float | None]) -> Dist:
    """Summarize ``values``, skipping the ones that are ``None``.

    Dispersion, not the mean, was the finding of the first experiment to
    use this package: a policy that leaves the median alone can still
    produce runaway trials.
    """
    known = [value for value in values if value is not None]
    if not known:
        return Dist()
    return Dist(
        n=len(known),
        mean=round(statistics.fmean(known), 1),
        sd=round(statistics.stdev(known), 1) if len(known) > 1 else 0.0,
        median=round(statistics.median(known), 1),
        minimum=min(known),
        maximum=max(known),
    )


@dataclasses.dataclass(slots=True)
class Tally:
    """Rates and distributions for one cell.

    ``unknowns`` counts, per check, the trials that could not answer it --
    a record written before tool outcomes were captured, asked about
    retries. Those trials leave the denominator rather than counting as
    misses.
    """

    cell: str
    n: int = 0
    hits: dict[str, int] = dataclasses.field(default_factory=dict)
    unknowns: dict[str, int] = dataclasses.field(default_factory=dict)
    mean_turns: float = 0.0
    mean_secs: float = 0.0
    turns: Dist = dataclasses.field(default_factory=Dist)
    secs: Dist = dataclasses.field(default_factory=Dist)
    retries: Dist = dataclasses.field(default_factory=Dist)

    def rate(self, name: str) -> float | None:
        """Fraction of the trials that could answer ``name``.

        ``None`` when none could: an empty cell, or a check whose material
        no record carries.
        """
        answered = self.n - self.unknowns.get(name, 0)
        if not answered:
            return None
        return self.hits.get(name, 0) / answered

    @property
    def secs_per_turn(self) -> float | None:
        """Mean seconds per tool call, or ``None`` without both.

        Separates "more round-trips" from "slower round-trips", which a
        wall-clock mean alone conflates.
        """
        if not self.turns.mean or self.secs.mean is None:
            return None
        return round(self.secs.mean / self.turns.mean, 2)


def tally(
    cell: str, trials: Iterable[TrialRecord], checks: Sequence[Check]
) -> Tally:
    result = Tally(
        cell=cell,
        hits={check.name: 0 for check in checks},
        unknowns={check.name: 0 for check in checks},
    )
    turns: list[float | None] = []
    secs: list[float | None] = []
    retries: list[float | None] = []
    for record in trials:
        result.n += 1
        turns.append(float(len(record.tool_calls)))
        secs.append(record.elapsed_s)
        counted = retry_count(record)
        retries.append(None if counted is None else float(counted))
        for check in checks:
            outcome = check.test(record)
            if outcome is None:
                result.unknowns[check.name] += 1
            elif outcome:
                result.hits[check.name] += 1
    result.turns = summarize(turns)
    result.secs = summarize(secs)
    result.retries = summarize(retries)
    result.mean_turns = result.turns.mean or 0.0
    result.mean_secs = result.secs.mean or 0.0
    return result
def render(tallies: Sequence[Tally], checks: Sequence[Check]) -> str:
    """A fixed-width table. N is always shown; rates are never shown alone."""
    headers = ["cell", "n", *(c.name for c in checks), "turns", "secs"]
    widths = [max(len(h), 6) for h in headers]
    widths[0] = max(len(t.cell) for t in tallies) if tallies else 4

    def row(cells: Sequence[str]) -> str:
        return "  ".join(
            c.rjust(w) for c, w in zip(cells, widths, strict=True)
        )

    lines = [row(headers), row(["-" * w for w in widths])]
    for item in tallies:
        rates = []
        for check in checks:
            value = item.rate(check.name)
            rates.append("-" if value is None else f"{100 * value:.0f}%")
        lines.append(
            row(
                [
                    item.cell,
                    str(item.n),
                    *rates,
                    f"{item.mean_turns}",
                    f"{item.mean_secs}",
                ]
            )
        )
    return "\n".join(lines)


# -- ready-made checks ----------------------------------------------------


def succeeded() -> Check:
    return Check("ok", lambda record: record.ok)


def response_contains(expected: str, *, ignore_commas: bool = True) -> Check:
    """Does the answer contain ``expected``?

    ``ignore_commas`` strips thousands separators first, so a model that
    replies "40,935.89" still matches "40935.89".
    """

    def test(record: TrialRecord) -> bool:
        text = record.response or ""
        if ignore_commas:
            text = text.replace(",", "")
        return expected in text

    return Check("correct", test)


def called_tool(name: str) -> Check:
    """Did the run call ``name`` at least once?"""
    return Check(
        f"used {name}", lambda record: name in set(record.call_names)
    )


def invented_tool_name(known: Iterable[str]) -> Check:
    """Did the run call a tool this room does not have?

    The check that motivated this package. It is only answerable because
    tool names are read from the run's message history rather than from tool
    spans, which never exist for a name that failed to resolve.
    """
    allowed = set(known)

    def test(record: TrialRecord) -> bool:
        return any(name not in allowed for name in record.call_names)

    return Check("bad tool", test)


def invalid_argument(
    argument: str, valid: Iterable[str | None]
) -> Check:
    """Did any call pass ``argument`` a value outside ``valid``?"""
    allowed = set(valid)

    def test(record: TrialRecord) -> bool:
        for call in record.tool_calls:
            parsed = call.parsed_arguments
            if argument in parsed and parsed[argument] not in allowed:
                return True
        return False

    return Check(f"bad {argument}", test)


# -- failure shapes -------------------------------------------------------


def classify_name(name: str | None) -> str:
    """How a bogus tool name is malformed.

    The distinction earned its keep: a model that emits
    ``run_python(environment_name="x")`` as a *name* has fallen out of
    structured tool calling, which is a different failure from one that
    prefixes a real name with a capability id.
    """
    if name is None:
        return "missing"
    if "(" in name or "<tool_call" in name:
        return "garbled-call"
    if ":" in name:
        return "namespaced"
    return "other"


def bad_name_shapes(
    records: Iterable[TrialRecord], known: Iterable[str]
) -> dict[str, int]:
    """Count bogus names by shape, across ``records``."""
    allowed = set(known)
    counts: collections.Counter[str] = collections.Counter()
    for record in records:
        for name in record.call_names:
            if name not in allowed:
                counts[classify_name(name)] += 1
    return dict(counts)


# -- distributions --------------------------------------------------------

#: Metrics ``render_distributions`` can draw.
DISTRIBUTIONS = ("turns", "secs", "retries")

_UNKNOWN_DISTRIBUTION = "field must be one of: turns, secs, retries"


def render_distributions(
    tallies: Sequence[Tally], field: str = "turns"
) -> str:
    """A fixed-width table of one metric's distribution, per cell.

    Kept apart from ``render`` on purpose: the rates table is already at
    its width, and a tail only shows up in a standard deviation and a
    maximum standing next to a median.
    """
    if field not in DISTRIBUTIONS:
        raise ValueError(_UNKNOWN_DISTRIBUTION)

    headers = [field, "n", "mean", "sd", "median", "min", "max"]
    widths = [max(len(h), 6) for h in headers]
    widths[0] = max((len(t.cell) for t in tallies), default=4)
    widths[0] = max(widths[0], len(field))

    def row(cells: Sequence[str]) -> str:
        return "  ".join(
            c.rjust(w) for c, w in zip(cells, widths, strict=True)
        )

    def cell_values(dist: Dist) -> list[str]:
        return [
            "-" if value is None else f"{value:g}"
            for value in (
                dist.mean,
                dist.sd,
                dist.median,
                dist.minimum,
                dist.maximum,
            )
        ]

    lines = [row(headers), row(["-" * w for w in widths])]
    for item in tallies:
        dist = getattr(item, field)
        lines.append(
            row([item.cell, str(dist.n), *cell_values(dist)])
        )
    return "\n".join(lines)


# -- outcomes -------------------------------------------------------------

#: Every ``ModelRetry`` reaches the model through
#: ``RetryPromptPart.model_response``, which appends this sentence to
#: whatever the retry said. It is the only reliable marker that a
#: ``tool_call_response`` is a retry rather than a success: both parts
#: serialize under that one type, with no discriminator between them.
RETRY_SUFFIX = "Fix the errors and try again."


def _is_wrapped_error(text: str) -> bool:
    """Does a payload carry a tool's own failure?

    A ``ToolReturnPart`` whose ``outcome`` is ``'failed'`` is serialized as
    an ``{'error': ...}`` mapping -- alone, or first in a list alongside
    file references. The ``outcome`` field itself never reaches the
    history, so this is the only trace of it.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return isinstance(parsed, dict) and "error" in parsed


def classify_result(text: str) -> str:
    """What one ``tool_call_response`` says happened.

    A retry is a round-trip the model spent and recovered from, which is
    invisible in ``ok`` and in ``correct``: the run still succeeds. Which
    *kind* of retry matters, because an invented name and a bounced
    ``load_capability`` are different behaviours.
    """
    stripped = text.rstrip()
    if stripped.endswith(RETRY_SUFFIX):
        if "Unknown tool name" in stripped:
            return "retry-unknown-tool"
        if "already available" in stripped:
            return "retry-already-available"
        if "validation error" in stripped:
            return "retry-validation"
        return "retry-other"
    if _is_wrapped_error(stripped):
        return "tool-error"
    return "ok"


def retry_count(record: TrialRecord) -> int | None:
    """How many calls came back as a retry.

    ``None`` -- not ``0`` -- when the record does not carry outcomes at
    all, which is every record written before they were captured.
    """
    if record.tool_results is None:
        return None
    return sum(
        1
        for result in record.tool_results
        if classify_result(result.text).startswith("retry-")
    )


def call_count(record: TrialRecord, name: str) -> int:
    """How many times ``name`` was called."""
    return sum(1 for called in record.call_names if called == name)


def retried() -> Check:
    """Did any call come back as a retry?"""

    def test(record: TrialRecord) -> bool | None:
        counted = retry_count(record)
        return None if counted is None else counted > 0

    return Check("retry", test)


def tool_errored() -> Check:
    """Did a tool run and hand back a failure?"""

    def test(record: TrialRecord) -> bool | None:
        if record.tool_results is None:
            return None
        return any(
            classify_result(result.text) == "tool-error"
            for result in record.tool_results
        )

    return Check("tool error", test)


def called_repeatedly(name: str) -> Check:
    """Was ``name`` called more than once in a single trial?

    Worth asking of a tool the room prompt describes on top of the
    capability that owns it: a duplicated description has already been
    measured driving a tool to be called when it had nothing to do.
    """
    return Check(
        f"{name} x2+", lambda record: call_count(record, name) > 1
    )


def result_shapes(records: Iterable[TrialRecord]) -> dict[str, int]:
    """Count response shapes other than ``ok``, across ``records``."""
    counts: collections.Counter[str] = collections.Counter()
    for record in records:
        for result in record.tool_results or ():
            shape = classify_result(result.text)
            if shape != "ok":
                counts[shape] += 1
    return dict(counts)
