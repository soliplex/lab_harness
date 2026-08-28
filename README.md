# `soliplex_lab_harness`

Reusable machinery for running experiments against a soliplex installation.

See the sibling [lab_bench](https://github.com/soliplex/lab_bench) repository
for how it is intended to be used: this package is the part that does not
change per experiment, and a jig depends on a pinned version of it.

## Installing

There is no PyPI release. Pin a tag:

```toml
dependencies = [
    "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@v0.1.1",
]
```

Release notes live in
[GitHub Releases](https://github.com/soliplex/lab_harness/releases); there is
no `CHANGELOG.md` to keep in sync.

## What is here

| module | does |
| --- | --- |
| `collect` | captures the tool calls a run actually made |
| `records` | one record per trial, appended to JSONL |
| `drive` | runs one turn against an installation, in process |
| `scoring` | turns records into rates, with pluggable checks |
| `environs` | builds one virtualenv per code-axis value |

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

## Code axes install; they do not check out

Each code-axis value is a `Pin` -- an exact released version, or a git ref --
installed into its own virtualenv:

```python
from pathlib import Path

from soliplex_lab_harness import environs

released = environs.Pin(name="v078", version="0.78.1")
from_ref = environs.Pin(
    name="tip",
    url="https://github.com/soliplex/soliplex",
    ref="v0.78.1",
)

env = environs.build(
    released,
    Path("envs/v078"),
    extra_requirements=(
        "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@v0.1.1",
    ),
)
print(env.python)        # interpreter to drive trials with
print(env.metadata())    # stamp this onto every record from this arm
```

`extra_requirements` normally includes this package, because a trial is
driven in process: the harness has to be importable in the environment that
holds the software under test. Passing it explicitly also forces the
experiment to record *which harness version measured the run*.

`build()` resolves a tag or branch to a **commit sha** via `git ls-remote`
and puts it in `metadata()`. Recording a tag records nothing -- tags can be
moved, and branches certainly are.

### Overlays: isolating co-landed changes

A release bundles every change in it. To attribute an effect to one of them
you need an arm that is "ref A, but with this one file from ref B":

```python
env = environs.build(
    environs.Pin(name="v077skill", version="0.77.2"),
    Path("envs/v077skill"),
    overlays=[
        environs.Overlay(
            source=Path("overlays/SKILL.md"),          # committed with the experiment
            destination="soliplex/skills/bwrap_sandbox/SKILL.md",
            note="SKILL.md as of v0.78.1",
        )
    ],
    extra_requirements=(
        "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@v0.1.1",
    ),
)
```

The overlay refuses to *create* a file: a typo in `destination` would
otherwise silently add something the software never shipped, and the arm
would measure software nobody runs. It also lands in `metadata()`, so the
arm describes itself.

### Everything shells out through an injected runner

`build()` and `Pin.resolve_ref()` take a `runner`, so the whole module is
testable without a network or a real `uv`. The default runner raises
`CommandFailed` with the captured stderr.

## Caveats

- `drive.run_trial` adjusts the process working directory and environment
  around the call, so it is **not thread-safe**. Trials run sequentially.
- `drive.install_collector` configures logfire, which is process-global.
  Call it once.
- Instrumentation is local only (`send_to_logfire=False`): nothing leaves
  the machine.
- `environs.build` shells out to `uv`, which must be on `PATH`.
- Confirm that behavior-affecting **non-Python assets are packaged**. An
  editable checkout exposes every file in the tree; an installed
  distribution exposes only what its packaging includes, so an unpackaged
  asset means an installed arm measures different software than a checkout
  would. soliplex ships `SKILL.md` via `MANIFEST.in`'s `global-include`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check
```
