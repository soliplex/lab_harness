# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors
should read [README.md](README.md).

## Do not add dependencies

`pyproject.toml` says `dependencies = []` and that is load-bearing, not an
oversight.

`drive.py` contains `import soliplex` and `import logfire` inside functions.
The obvious "fix" is to declare them as dependencies. **Do not.** `soliplex`
is the software under test: an experiment installs a different pinned version
of it per code-axis value, so a pin here would fight the axis it exists to
measure. `logfire` arrives with soliplex.

If a genuinely new dependency is needed, it is a design decision, not
housekeeping.

## Do not "simplify" the collector

`collect.py` reads the agent run span's `pydantic_ai.all_messages` attribute
rather than harvesting spans named `running tool`. This looks roundabout and
is not.

`ToolManager._resolve_tool` rejects an unknown tool name *before*
`wrap_tool_execute` opens a span, so a tool name the model **invented**
produces no span at all. A span-harvesting collector reports a clean sheet no
matter what the model did, and the resulting null looks like a finding. That
failure mode is the reason this package exists.

## Conventions

- 79 columns, single-line imports, ruff rule sets `F, E, B, UP, I, PD, TRY,
  PT`. Run `uv run ruff check` and `uv run pytest`.
- Exception classes carry their own messages -- see `environs.py` -- rather
  than suppressing `TRY003`.
- No re-export-only `__init__.py`. Callers import the modules.
- Tests must not need a network, a real `uv`, or an LLM. Subprocess work goes
  through an injected `runner`; instrumentation is faked with a stub span.
  The suite runs in well under a second, and should stay that way.

## Releases

Release notes live in **GitHub Releases**. Do not add a `CHANGELOG.md`.

There is no PyPI release. Consumers pin a tag:

    soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@v0.2

### Tag convention

Follow soliplex, which has held to this across 170-odd release tags:

| tag | for |
| --- | --- |
| `v0.X` | the first release of a minor line |
| `v0.X.N` | a bugfix release on that line |
| `v0.X.N.M` | a packaging-only re-release of unchanged code |

`version` in `pyproject.toml` is the tag without the leading `v` -- soliplex
carries `version = "0.78"` at `v0.78`. So the first release of a line is
`v0.1`, **not** `v0.1.0`: do not invent a third component that the
convention does not use.

Any pin written into documentation has to be updated in the same commit that
changes the version, or the docs teach a pin that does not resolve.
