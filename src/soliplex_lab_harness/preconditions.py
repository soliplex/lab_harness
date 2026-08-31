"""Assert what only a recorded turn can establish.

Preconditions are checked *before* trials are spent, because a cell that
structurally cannot exhibit the behaviour under test yields a null that
reads as a real result. Every defect this kind of check has caught so far
was silent: the run completed and the table looked plausible. So these are
assertions, not notes.

**Nothing here drives a turn.** A check reads what a run already recorded.
Checks that need no turn belong with whatever they check, on the principle
of make-the-thing-then-verify-the-thing -- ``environs.verify_install``
compares an install against its own RECORD at build time, and raises there
rather than being reported here.

What a check *is* stays the caller's business: a check is any function that
returns ``Result`` values, because what is worth asserting differs entirely
between experiment sets. This module owns only the result record and the
two things anyone does with a list of them -- render it, or refuse to
continue.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """One named yes/no question, and how it came out for one cell."""

    cell: str
    what: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        tail = f" -- {self.detail}" if self.detail else ""
        return f"  {mark} {self.cell}: {self.what}{tail}"

    @property
    def summary(self) -> str:
        """The cell and question, without the pass/fail marker."""
        tail = f" -- {self.detail}" if self.detail else ""
        return f"{self.cell}: {self.what}{tail}"


class Failed(Exception):
    """A run would have measured something other than it claims."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        body = "\n".join(f"  - {failure}" for failure in self.failures)
        super().__init__(
            f"{len(self.failures)} precondition(s) failed:\n{body}"
        )


def assert_ok(results: Sequence[Result]) -> None:
    """Raise ``Failed`` naming every check that did not hold."""
    failures = [result.summary for result in results if not result.ok]
    if failures:
        raise Failed(failures)


def render(results: Sequence[Result]) -> str:
    """Every result, then a one-line verdict.

    An empty list says so rather than reporting that everything holds.
    Nothing checked is not the same as nothing wrong, and this module
    exists because that distinction gets lost.
    """
    if not results:
        return "no preconditions were checked"

    lines = [str(result) for result in results]
    failed = [result for result in results if not result.ok]
    if failed:
        lines.append(f"\n{len(failed)} precondition(s) failed")
    else:
        lines.append(f"\nall {len(results)} preconditions hold")
    return "\n".join(lines)
