"""Build one virtualenv per code-axis value.

A code axis does not check the software out -- it **installs** it. Each
axis value is a :class:`Pin`: either an exact released version or a git ref,
materialized as a pinned requirement in its own virtualenv.

Why not worktrees: an experiment then never touches a checkout it does not
own, so there is nothing to dirty or restore; an arm reduces to a
requirement string small enough to record on an issue; and it measures the
software *as shipped* rather than as an editable working tree.

Every subprocess goes through an injected ``runner``, so the whole module is
exercisable without a network or a real uv.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

#: ``(argv, cwd) -> stdout``. Raises on failure.
Runner = Callable[[Sequence[str], pathlib.Path | None], str]

_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_BIN = "Scripts" if os.name == "nt" else "bin"


class EnvironmentBuildError(Exception):
    """Something about a pin or an environment is unusable."""


class BadPin(EnvironmentBuildError):
    def __init__(self, name: str, problem: str):
        self.name = name
        self.problem = problem
        super().__init__(f"pin {name!r}: {problem}")


class RefNotFound(EnvironmentBuildError):
    def __init__(self, name: str, ref: str, url: str | None):
        self.name = name
        self.ref = ref
        self.url = url
        super().__init__(f"pin {name!r}: {ref!r} not found at {url}")


class CommandFailed(EnvironmentBuildError):
    def __init__(self, argv: Sequence[str], returncode: int, stderr: str):
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"{' '.join(argv)} failed ({returncode}): {stderr.strip()}"
        )


class NoSitePackages(EnvironmentBuildError):
    def __init__(self, root: pathlib.Path):
        self.root = root
        super().__init__(f"no site-packages under {root}; was it built?")


class OverlaySourceMissing(EnvironmentBuildError):
    def __init__(self, source: pathlib.Path):
        self.source = source
        super().__init__(f"overlay source missing: {source}")


class OverlayDestinationMissing(EnvironmentBuildError):
    """An overlay replaces shipped content; it never adds a new file.

    A typo in ``destination`` would otherwise silently create a file the
    software never had, which measures something nobody ships.
    """

    def __init__(self, destination: str, site: pathlib.Path):
        self.destination = destination
        self.site = site
        super().__init__(
            f"overlay destination {destination!r} does not exist under "
            f"{site}; an overlay replaces shipped content, it does not add "
            "new files"
        )


class NoRecord(EnvironmentBuildError):
    def __init__(self, distribution: str, site: pathlib.Path):
        self.distribution = distribution
        self.site = site
        super().__init__(
            f"no {distribution}-*.dist-info/RECORD under {site}"
        )


class InstallDiverged(EnvironmentBuildError):
    """Installed files do not match the distribution's own RECORD.

    Almost always means something wrote into the environment after
    installation. The case this was built for: writing through a hardlink
    that uv shares with its cache, which silently rewrites the same file in
    every sibling environment holding that distribution.
    """

    def __init__(
        self, distribution: str, unexpected: Sequence[str]
    ):
        self.distribution = distribution
        self.unexpected = list(unexpected)
        listed = ", ".join(self.unexpected[:5])
        more = (
            f" (and {len(self.unexpected) - 5} more)"
            if len(self.unexpected) > 5
            else ""
        )
        super().__init__(
            f"{distribution} install diverges from its RECORD at: "
            f"{listed}{more}"
        )


class OverlayNotApplied(EnvironmentBuildError):
    """An overlay was declared but the file still matches RECORD.

    So the arm is not the arm it claims to be, and would measure the
    unmodified software while reporting otherwise.
    """

    def __init__(self, destinations: Sequence[str]):
        self.destinations = list(destinations)
        super().__init__(
            "overlay declared but file unchanged: "
            + ", ".join(self.destinations)
        )


def run(argv: Sequence[str], cwd: pathlib.Path | None = None) -> str:
    """Default runner: run ``argv``, return stdout, raise on failure."""
    completed = subprocess.run(  # noqa: S603 -- argv is built, never a string
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CommandFailed(argv, completed.returncode, completed.stderr)
    return completed.stdout


@dataclasses.dataclass(frozen=True, slots=True)
class Pin:
    """One code-axis value: what to install, and what to call it.

    Give either ``version`` (a released version) or ``url`` plus ``ref``
    (any other commit-ish). ``name`` is the short label that appears in cell
    names and in a record's metadata.
    """

    name: str
    package: str = "soliplex"
    version: str | None = None
    url: str | None = None
    ref: str | None = None
    extras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from_version = self.version is not None
        from_git = self.url is not None or self.ref is not None
        if from_version and from_git:
            raise BadPin(self.name, "give version or url+ref, not both")
        if not from_version and not from_git:
            raise BadPin(self.name, "needs a version or a url+ref")
        if from_git and not (self.url and self.ref):
            raise BadPin(self.name, "a git pin needs both url and ref")

    @property
    def _named(self) -> str:
        if not self.extras:
            return self.package
        return f"{self.package}[{','.join(self.extras)}]"

    def requirement(self) -> str:
        """The PEP 508 requirement string this pin installs."""
        if self.version is not None:
            return f"{self._named}=={self.version}"
        return f"{self._named} @ git+{self.url}@{self.ref}"

    def resolve_ref(self, runner: Runner = run) -> str | None:
        """The concrete commit sha for this pin, when it has one.

        A version pin has no sha. A ref that is already a full sha is
        returned unchanged; anything else is looked up with ``git
        ls-remote``, because recording a tag or branch records nothing --
        tags can move and branches certainly do.
        """
        if self.ref is None:
            return None
        if _SHA.match(self.ref):
            return self.ref
        out = runner(["git", "ls-remote", str(self.url), self.ref], None)
        for line in out.splitlines():
            sha, _, _name = line.partition("\t")
            if _SHA.match(sha.strip()):
                return sha.strip()
        raise RefNotFound(self.name, self.ref, self.url)


@dataclasses.dataclass(frozen=True, slots=True)
class Overlay:
    """One file held at a different state than the rest of the install.

    The mechanism for isolating co-landed changes: "ref A, but with this
    file from ref B". ``source`` is a file committed alongside the
    experiment, so the arm describes itself and no branch has to be invented
    upstream for a state nobody shipped.
    """

    source: pathlib.Path
    destination: str
    note: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Environment:
    """A built virtualenv holding one code-axis value."""

    pin: Pin
    root: pathlib.Path
    resolved_ref: str | None = None
    overlays: tuple[Overlay, ...] = ()

    @property
    def python(self) -> pathlib.Path:
        return self.root / _BIN / "python"

    def site_packages(self) -> pathlib.Path:
        """Where the installed distribution lives.

        Looked up in the environment itself rather than guessed from a
        version number.
        """
        pattern = "lib/python*/site-packages"
        for candidate in sorted(self.root.glob(pattern)):
            if candidate.is_dir():
                return candidate
        raise NoSitePackages(self.root)

    def metadata(self) -> dict[str, Any]:
        """What identifies this arm, for a record's ``metadata``."""
        data: dict[str, Any] = {
            "axis_value": self.pin.name,
            "requirement": self.pin.requirement(),
        }
        if self.resolved_ref is not None:
            data["resolved_ref"] = self.resolved_ref
        if self.overlays:
            data["overlays"] = [
                {"destination": o.destination, "note": o.note}
                for o in self.overlays
            ]
        return data


def venv_argv(root: pathlib.Path, python: str | None) -> list[str]:
    argv = ["uv", "venv"]
    if python is not None:
        argv += ["--python", python]
    return [*argv, str(root)]


def install_argv(
    root: pathlib.Path, requirements: Sequence[str]
) -> list[str]:
    return [
        "uv",
        "pip",
        "install",
        "--python",
        str(root / _BIN / "python"),
        *requirements,
    ]


def build(
    pin: Pin,
    root: pathlib.Path,
    *,
    extra_requirements: Iterable[str] = (),
    overlays: Iterable[Overlay] = (),
    python: str | None = None,
    recreate: bool = False,
    resolve: bool = True,
    runner: Runner = run,
) -> Environment:
    """Create ``root`` as a virtualenv with ``pin`` installed.

    ``extra_requirements`` normally includes this package: a trial is driven
    in process, so the harness has to be importable in the environment that
    holds the software under test. Passing it explicitly also forces the
    experiment to record which harness version measured the run.
    """
    if recreate and root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    resolved = pin.resolve_ref(runner) if resolve else None

    runner(venv_argv(root, python), None)
    requirements = [pin.requirement(), *extra_requirements]
    runner(install_argv(root, requirements), None)

    applied = tuple(overlays)
    environment = Environment(
        pin=pin, root=root, resolved_ref=resolved, overlays=applied
    )
    if applied:
        apply_overlays(environment, applied)
    return environment


def apply_overlays(
    environment: Environment, overlays: Iterable[Overlay]
) -> None:
    """Copy each overlay into the built environment.

    Refuses to create a file that the installed distribution does not
    already have: an overlay is meant to *replace* shipped content, and a
    typo in ``destination`` would otherwise silently add a file the software
    never had, which measures something nobody ships.

    The target is **unlinked before writing**, which is not optional. uv
    installs by hardlinking out of its cache, so a freshly built
    environment's files typically share an inode with the cache and with
    every other environment holding the same distribution. Writing through
    that hardlink -- as ``shutil.copyfile`` alone does -- mutates the shared
    inode: it silently rewrites the same file in sibling environments *and*
    poisons the cache for every later install of that version. The damage
    is invisible until something compares an install against its own
    ``RECORD``.
    """
    site = environment.site_packages()
    for overlay in overlays:
        if not overlay.source.is_file():
            raise OverlaySourceMissing(overlay.source)
        target = site / overlay.destination
        if not target.is_file():
            raise OverlayDestinationMissing(overlay.destination, site)
        target.unlink()
        shutil.copyfile(overlay.source, target)


# -- verifying an install ------------------------------------------------
#
# A built environment is only trustworthy if it still matches what was
# installed. Two ways it can stop matching, both silent:
#
#   * something wrote into it -- notably an overlay applied through a
#     hardlink uv shares with its cache, which rewrites the same file in
#     every sibling environment holding that distribution
#   * an overlay was declared and did not land, so the arm measures the
#     unmodified software while reporting that it does not
#
# Both produce plausible numbers rather than errors, so they are worth
# asserting before spending trials.


def _record_path(
    environment: Environment, distribution: str
) -> pathlib.Path:
    site = environment.site_packages()
    for candidate in sorted(site.glob(f"{distribution}-*.dist-info/RECORD")):
        return candidate
    raise NoRecord(distribution, site)


def _file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return f"sha256={encoded}"


def record_entries(
    environment: Environment, distribution: str
) -> dict[str, str]:
    """Map site-relative path -> expected hash, from the RECORD file.

    Entries with no hash (RECORD itself, and anything the installer chose
    not to checksum) are skipped, as are paths outside site-packages and
    compiled bytecode, none of which say anything about tampering.
    """
    entries: dict[str, str] = {}
    text = _record_path(environment, distribution).read_text(
        encoding="utf-8"
    )
    for line in text.splitlines():
        name, _, rest = line.partition(",")
        expected = rest.split(",")[0] if rest else ""
        if not name or not expected:
            continue
        if name.startswith("..") or name.endswith(".pyc"):
            continue
        entries[name] = expected
    return entries


def diverged(
    environment: Environment, distribution: str
) -> dict[str, str | None]:
    """Site-relative paths whose contents no longer match RECORD.

    The value is the on-disk hash, or ``None`` when the file is missing.
    """
    site = environment.site_packages()
    out: dict[str, str | None] = {}
    for name, expected in record_entries(environment, distribution).items():
        target = site / name
        if not target.is_file():
            out[name] = None
        elif _file_hash(target) != expected:
            out[name] = _file_hash(target)
    return out


def verify_install(
    environment: Environment,
    distribution: str | None = None,
    *,
    expect_modified: Iterable[str] = (),
) -> dict[str, str | None]:
    """Assert the install matches RECORD except where an overlay says so.

    ``expect_modified`` is the set of site-relative paths an overlay was
    meant to change; by default it is taken from the environment's own
    overlays, so the ordinary call is ``verify_install(env)``.

    Raises :class:`InstallDiverged` for any *other* changed file, and
    :class:`OverlayNotApplied` if a declared overlay left its target
    untouched. Returns the divergences it accepted.
    """
    distribution = distribution or environment.pin.package
    expected_set = set(expect_modified) or {
        overlay.destination for overlay in environment.overlays
    }
    found = diverged(environment, distribution)

    unexpected = sorted(set(found) - expected_set)
    if unexpected:
        raise InstallDiverged(distribution, unexpected)

    missing = sorted(expected_set - set(found))
    if missing:
        raise OverlayNotApplied(missing)

    return found
