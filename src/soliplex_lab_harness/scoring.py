"""Turn trial records into rates.

Two habits are baked in, both learned the hard way:

* rates, not outcomes -- one run tells you nothing about a stochastic system
* report N alongside every rate, because at N=20 a one- or two-run
  difference is not a result
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

from .records import TrialRecord

Predicate = Callable[[TrialRecord], bool]


@dataclasses.dataclass(frozen=True, slots=True)
class Check:
    """A named per-trial yes/no question."""

    name: str
    test: Predicate


@dataclasses.dataclass(slots=True)
class Tally:
    cell: str
    n: int = 0
    hits: dict[str, int] = dataclasses.field(default_factory=dict)
    mean_turns: float = 0.0
    mean_secs: float = 0.0

    def rate(self, name: str) -> float | None:
        """Fraction of trials satisfying ``name``, or None when n == 0."""
        if not self.n:
            return None
        return self.hits.get(name, 0) / self.n


def tally(
    cell: str, trials: Iterable[TrialRecord], checks: Sequence[Check]
) -> Tally:
    result = Tally(cell=cell, hits={check.name: 0 for check in checks})
    turns = 0
    secs = 0.0
    for record in trials:
        result.n += 1
        turns += len(record.tool_calls)
        secs += record.elapsed_s
        for check in checks:
            if check.test(record):
                result.hits[check.name] += 1
    if result.n:
        result.mean_turns = round(turns / result.n, 1)
        result.mean_secs = round(secs / result.n, 1)
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
