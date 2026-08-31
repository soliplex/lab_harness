"""One record per trial, appended to a JSONL file."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

from .collect import ToolCall
from .collect import ToolResult


@dataclasses.dataclass(slots=True)
class TrialRecord:
    """What one run of one cell produced.

    ``metadata`` is deliberately open: an experiment stamps whatever
    identifies the arm -- the resolved commit sha under test, the exact
    model id, the endpoint -- without this class needing to know about any
    of it. A result whose arm cannot be identified later is not a result.

    ``tool_results`` is ``None`` when outcomes were not captured, and ``[]``
    when they were captured and there were none. The distinction is
    load-bearing: records written before outcomes were collected cannot say
    whether a call failed, and reporting "no retries" for them would be a
    null that reads as a result. Every consumer that counts outcomes returns
    ``None`` rather than ``0`` for such a record.
    """

    cell: str
    trial: int
    elapsed_s: float
    ok: bool
    error: str | None = None
    response: str | None = None
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    tool_results: list[ToolResult] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["tool_calls"] = [
            {
                "name": call.name,
                "arguments": call.arguments,
                "id": call.id,
            }
            for call in self.tool_calls
        ]
        if self.tool_results is None:
            data["tool_results"] = None
        else:
            data["tool_results"] = [
                {
                    "name": result.name,
                    "id": result.id,
                    "result": result.result,
                }
                for result in self.tool_results
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialRecord:
        fields = dict(data)
        fields["tool_calls"] = [
            ToolCall(
                name=c.get("name"),
                arguments=c.get("arguments"),
                id=c.get("id"),
            )
            for c in fields.get("tool_calls") or ()
        ]
        raw_results = fields.get("tool_results")
        fields["tool_results"] = (
            None
            if raw_results is None
            else [
                ToolResult(
                    name=r.get("name"),
                    id=r.get("id"),
                    result=r.get("result"),
                )
                for r in raw_results
            ]
        )
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in fields.items() if k in known})

    @property
    def call_names(self) -> list[str | None]:
        return [call.name for call in self.tool_calls]


def append(path: pathlib.Path, record: TrialRecord) -> None:
    """Append one record, creating the file and its parent if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict()) + "\n")


def read(path: pathlib.Path) -> list[TrialRecord]:
    return list(iter_read(path))


def iter_read(path: pathlib.Path) -> Iterator[TrialRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield TrialRecord.from_dict(json.loads(line))


def completed(path: pathlib.Path) -> int:
    """How many records ``path`` already holds; 0 when it does not exist."""
    return len(read(path)) if path.exists() else 0


def top_up(
    path: pathlib.Path,
    trials: int,
    run: Callable[[int], TrialRecord],
    *,
    on_trial: Callable[[TrialRecord], None] | None = None,
) -> int:
    """Top ``path`` up to ``trials`` records, and say how many ran.

    ``trials`` is a **target, not a count**. Records already in the file
    count toward it, so a smoke trial can be run and verified before
    extending to the full N without discarding it, and an interrupted run
    resumes instead of restarting or double-counting.

    A file that already holds enough is success, not an error: nothing runs
    and ``0`` comes back.

    Each record is appended as it arrives, so an interrupt loses at most one
    trial. ``on_trial`` runs only after the record is on disk; progress
    reporting stays the caller's business, because the line worth printing
    is usually particular to the experiment.

    ``run`` is handed the trial index. Trials are sequential by design --
    ``drive.run_trial`` adjusts the process cwd and environment, so running
    them concurrently is not safe.
    """
    done = completed(path)
    if done >= trials:
        return 0
    for trial in range(done, trials):
        record = run(trial)
        append(path, record)
        if on_trial is not None:
            on_trial(record)
    return trials - done
