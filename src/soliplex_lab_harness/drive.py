"""Run one trial against a soliplex installation, in process.

``soliplex`` and ``logfire`` are imported lazily and are deliberately *not*
dependencies of this package: soliplex is the software under test, pinned
per code-axis value by the experiment, so a pin here would fight the axis.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import pathlib
import time
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any

from . import records
from .collect import RunCollector

_MISSING = (
    "{name} is not importable. Install the software under test into this "
    "environment -- an experiment pins it per code-axis value, e.g. "
    "'soliplex==0.78.1' or 'soliplex @ git+<url>@<sha>'."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """Where and what to run.

    ``cwd`` matters: a soliplex installation may carry cwd-relative sqlite
    URIs, so each cell normally runs with its own directory.
    """

    installation: pathlib.Path
    room_id: str
    cwd: pathlib.Path | None = None
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)


def install_collector(collector: RunCollector | None = None) -> RunCollector:
    """Wire a collector into local-only instrumentation.

    Call once per process: logfire configuration is global. Nothing is sent
    anywhere -- ``send_to_logfire=False`` -- so this stays usable offline and
    leaks no run content.
    """
    try:
        import logfire
    except ImportError as exc:  # pragma: no cover -- environment-dependent
        raise RuntimeError(_MISSING.format(name="logfire")) from exc

    collector = collector if collector is not None else RunCollector()
    logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[collector],
    )
    logfire.instrument_pydantic_ai(include_content=True)
    return collector


@contextlib.contextmanager
def _in_directory(path: pathlib.Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    previous = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _with_env(extra: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in extra}
    os.environ.update(extra)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_installation(target: Target) -> Any:
    """Resolve a soliplex installation, secrets and environment included."""
    try:
        from soliplex.cli import cli_util
    except ImportError as exc:
        raise RuntimeError(_MISSING.format(name="soliplex")) from exc

    installation = cli_util.get_installation(target.installation)
    installation.resolve_secrets()
    installation.resolve_environment()
    return installation


def run_trial(
    target: Target,
    prompt: str,
    *,
    cell: str,
    trial: int,
    collector: RunCollector,
    installation: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> records.TrialRecord:
    """Run one turn and return its record.

    Not thread-safe: the process cwd and environment are adjusted around the
    call. Trials are meant to run sequentially.
    """
    try:
        from soliplex.cli import ask as ask_module
        from soliplex.cli import cli_util
    except ImportError as exc:
        raise RuntimeError(_MISSING.format(name="soliplex")) from exc

    claims = cli_util._audit_claims()
    collector.reset()
    started = time.monotonic()
    error: str | None = None
    result: Any = None

    with _in_directory(target.cwd), _with_env(target.env):
        if installation is None:
            installation = load_installation(target)
        try:
            result = asyncio.run(
                ask_module._run_ask(
                    installation, target.room_id, prompt, claims
                )
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, not handled
            error = f"{type(exc).__name__}: {exc}"

    calls, _raw = collector.take()
    return records.TrialRecord(
        cell=cell,
        trial=trial,
        elapsed_s=round(time.monotonic() - started, 2),
        ok=bool(result is not None and result.ok),
        error=error or (result.error if result is not None else None),
        response=result.response if result is not None else None,
        tool_calls=calls,
        metadata=dict(metadata or {}),
    )
