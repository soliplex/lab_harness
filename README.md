# `soliplex_lab_harness`

Reusable machinery for running experiments against a soliplex installation.

See the sibling [lab_bench](https://github.com/soliplex/lab_bench) repository
for how it is intended to be used: this package is the part that does not
change per experiment, and a jig depends on a pinned version of it.

## What is here

| module | does |
| --- | --- |
| `collect` | captures the tool calls a run actually made |
| `records` | one record per trial, appended to JSONL |
| `drive` | runs one turn against an installation, in process |
| `scoring` | turns records into rates, with pluggable checks |

Import the modules, not names from the package root.

## The one non-obvious thing

An **invented tool name never produces a tool span.**
`ToolManager._resolve_tool` rejects an unknown name before
`wrap_tool_execute` opens one, so a harness that harvests spans named
`running tool` reports a clean sheet no matter what the model did -- and a
null result then looks like a finding.

`collect` reads the agent run span's `pydantic_ai.all_messages` attribute
instead. That carries the whole message history, and every tool-call part in
it is recorded under the name the model actually used. It is also what
Logfire renders as a run's "events" list.

## No dependencies, on purpose

`soliplex` is the software under test. An experiment pins it per code-axis
value -- `soliplex==0.78.1`, or `soliplex @ git+<url>@<sha>` -- so a pin here
would fight the axis. Both `soliplex` and `logfire` are imported lazily by
`drive`, which raises a pointed error if they are missing.

## Sketch

```python
from pathlib import Path

from soliplex_lab_harness import drive
from soliplex_lab_harness import records
from soliplex_lab_harness import scoring

collector = drive.install_collector()   # once per process
target = drive.Target(
    installation=Path("cells/before-gemma4/example/minimal.yaml"),
    room_id="bwrap_sandbox",
    cwd=Path("cells/before-gemma4"),
    env={"OPENAI_API_KEY": "unused-by-vllm"},
)
installation = drive.load_installation(target)

out = Path("results/before-gemma4.jsonl")
for trial in range(20):
    record = drive.run_trial(
        target,
        "What is the total order value for the Southeast region?",
        cell="before-gemma4",
        trial=trial,
        collector=collector,
        installation=installation,
        metadata={"ref": "v0.78.1", "model": "gemma4-26b"},
    )
    records.append(out, record)

checks = [
    scoring.succeeded(),
    scoring.response_contains("40935.89"),
    scoring.invented_tool_name({"run", "run_python", "list_environments"}),
]
result = scoring.tally("before-gemma4", records.read(out), checks)
print(scoring.render([result], checks))
```

`metadata` is open on purpose: stamp whatever identifies the arm -- resolved
commit sha, exact model id, endpoint. A result whose arm cannot be identified
afterwards is not a result.

## Caveats

- `drive.run_trial` adjusts the process working directory and environment
  around the call, so it is **not thread-safe**. Trials run sequentially.
- `drive.install_collector` configures logfire, which is process-global.
  Call it once.
- Instrumentation is local only (`send_to_logfire=False`): nothing leaves
  the machine.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check
```
