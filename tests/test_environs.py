import pathlib

import pytest

from soliplex_lab_harness import environs

SHA = "a" * 40
URL = "https://github.com/soliplex/soliplex"


class FakeRunner:
    """Records argv lists and returns canned stdout."""

    def __init__(self, stdout=""):
        self.calls: list[list[str]] = []
        self.stdout = stdout

    def __call__(self, argv, cwd=None):
        self.calls.append(list(argv))
        return self.stdout


def make_venv(root: pathlib.Path) -> pathlib.Path:
    """Enough of a virtualenv layout for site_packages() to find."""
    site = root / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    return site


# -- Pin ------------------------------------------------------------------


def test_version_pin_requirement():
    pin = environs.Pin(name="v078", version="0.78.1")

    assert pin.requirement() == "soliplex==0.78.1"


def test_git_pin_requirement():
    pin = environs.Pin(name="tip", url=URL, ref=SHA)

    assert pin.requirement() == f"soliplex @ git+{URL}@{SHA}"


def test_extras_are_included():
    pin = environs.Pin(name="v078", version="0.78.1", extras=("google",))

    assert pin.requirement() == "soliplex[google]==0.78.1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"version": "0.78.1", "url": URL, "ref": SHA},
        {"url": URL},
        {"ref": SHA},
    ],
)
def test_bad_pins_are_rejected(kwargs):
    with pytest.raises(environs.BadPin):
        environs.Pin(name="bad", **kwargs)


def test_resolve_ref_passes_a_sha_through_without_asking_git():
    pin = environs.Pin(name="tip", url=URL, ref=SHA)
    runner = FakeRunner()

    assert pin.resolve_ref(runner) == SHA
    assert runner.calls == []


def test_resolve_ref_looks_up_a_tag():
    """A tag records nothing: tags move, so store the sha."""
    pin = environs.Pin(name="v078", url=URL, ref="v0.78.1")
    runner = FakeRunner(stdout=f"{SHA}\trefs/tags/v0.78.1\n")

    assert pin.resolve_ref(runner) == SHA
    assert runner.calls == [["git", "ls-remote", URL, "v0.78.1"]]


def test_resolve_ref_of_a_version_pin_is_none():
    pin = environs.Pin(name="v078", version="0.78.1")

    assert pin.resolve_ref(FakeRunner()) is None


def test_resolve_ref_raises_when_absent():
    pin = environs.Pin(name="nope", url=URL, ref="no-such-tag")

    with pytest.raises(environs.RefNotFound):
        pin.resolve_ref(FakeRunner(stdout=""))


# -- build ----------------------------------------------------------------


def test_build_creates_then_installs(tmp_path):
    pin = environs.Pin(name="v078", version="0.78.1")
    root = tmp_path / "envs" / "v078"
    runner = FakeRunner()

    environs.build(
        pin,
        root,
        extra_requirements=("soliplex-lab-harness==0.1.0",),
        runner=runner,
    )

    assert runner.calls[0] == ["uv", "venv", str(root)]
    assert runner.calls[1][:4] == [
        "uv",
        "pip",
        "install",
        "--python",
    ]
    assert runner.calls[1][-2:] == [
        "soliplex==0.78.1",
        "soliplex-lab-harness==0.1.0",
    ]


def test_build_honours_an_explicit_interpreter(tmp_path):
    runner = FakeRunner()

    environs.build(
        environs.Pin(name="v078", version="0.78.1"),
        tmp_path / "env",
        python="3.13",
        runner=runner,
    )

    assert runner.calls[0][:4] == ["uv", "venv", "--python", "3.13"]


def test_build_records_the_resolved_ref(tmp_path):
    pin = environs.Pin(name="tip", url=URL, ref="main")
    runner = FakeRunner(stdout=f"{SHA}\trefs/heads/main\n")

    env = environs.build(pin, tmp_path / "env", runner=runner)

    assert env.resolved_ref == SHA
    assert env.metadata()["resolved_ref"] == SHA


def test_build_recreate_removes_the_old_tree(tmp_path):
    root = tmp_path / "env"
    make_venv(root)
    (root / "stale").write_text("x", encoding="utf-8")

    environs.build(
        environs.Pin(name="v078", version="0.78.1"),
        root,
        recreate=True,
        runner=FakeRunner(),
    )

    assert not (root / "stale").exists()


# -- Environment ----------------------------------------------------------


def test_site_packages_is_discovered(tmp_path):
    root = tmp_path / "env"
    site = make_venv(root)
    env = environs.Environment(
        pin=environs.Pin(name="v078", version="0.78.1"), root=root
    )

    assert env.site_packages() == site


def test_site_packages_raises_when_unbuilt(tmp_path):
    env = environs.Environment(
        pin=environs.Pin(name="v078", version="0.78.1"),
        root=tmp_path / "missing",
    )

    with pytest.raises(environs.NoSitePackages):
        env.site_packages()


def test_metadata_identifies_the_arm():
    env = environs.Environment(
        pin=environs.Pin(name="v078", version="0.78.1"),
        root=pathlib.Path("/nowhere"),
    )

    assert env.metadata() == {
        "axis_value": "v078",
        "requirement": "soliplex==0.78.1",
    }


# -- overlays -------------------------------------------------------------


def test_overlay_replaces_shipped_content(tmp_path):
    root = tmp_path / "env"
    site = make_venv(root)
    shipped = site / "soliplex" / "skills" / "s"
    shipped.mkdir(parents=True)
    (shipped / "SKILL.md").write_text("old", encoding="utf-8")
    source = tmp_path / "new-SKILL.md"
    source.write_text("new", encoding="utf-8")
    env = environs.Environment(
        pin=environs.Pin(name="a", version="1.0"), root=root
    )
    overlay = environs.Overlay(
        source=source,
        destination="soliplex/skills/s/SKILL.md",
        note="from v0.78.1",
    )

    environs.apply_overlays(env, [overlay])

    assert (shipped / "SKILL.md").read_text(encoding="utf-8") == "new"


def test_overlay_refuses_to_add_a_new_file(tmp_path):
    """A typo would otherwise measure software nobody ships."""
    root = tmp_path / "env"
    make_venv(root)
    source = tmp_path / "f.md"
    source.write_text("x", encoding="utf-8")
    env = environs.Environment(
        pin=environs.Pin(name="a", version="1.0"), root=root
    )
    overlay = environs.Overlay(source=source, destination="typo/SKILL.md")

    with pytest.raises(environs.OverlayDestinationMissing):
        environs.apply_overlays(env, [overlay])


def test_overlay_requires_its_source(tmp_path):
    root = tmp_path / "env"
    make_venv(root)
    env = environs.Environment(
        pin=environs.Pin(name="a", version="1.0"), root=root
    )
    overlay = environs.Overlay(
        source=tmp_path / "absent.md", destination="anything"
    )

    with pytest.raises(environs.OverlaySourceMissing):
        environs.apply_overlays(env, [overlay])


def test_overlays_appear_in_metadata(tmp_path):
    root = tmp_path / "env"
    site = make_venv(root)
    (site / "a.md").write_text("old", encoding="utf-8")
    source = tmp_path / "b.md"
    source.write_text("new", encoding="utf-8")

    env = environs.build(
        environs.Pin(name="v077skill", version="0.77.2"),
        root,
        overlays=[
            environs.Overlay(
                source=source, destination="a.md", note="from v0.78.1"
            )
        ],
        runner=FakeRunner(),
    )

    assert env.metadata()["overlays"] == [
        {"destination": "a.md", "note": "from v0.78.1"}
    ]


# -- default runner -------------------------------------------------------


def test_run_raises_command_failed_on_nonzero():
    with pytest.raises(environs.CommandFailed) as caught:
        environs.run(["python3", "-c", "import sys; sys.exit(3)"])

    assert caught.value.returncode == 3


def test_run_returns_stdout():
    out = environs.run(["python3", "-c", "print('hi')"])

    assert out.strip() == "hi"


def test_overlay_does_not_write_through_a_hardlink(tmp_path):
    """The bug that made an overlay corrupt a sibling environment.

    uv installs by hardlinking out of its cache, so a fresh environment's
    files share an inode with the cache and with every other environment
    holding the same distribution. Writing through that link mutates all of
    them at once. The overlay must break the link first.
    """
    root = tmp_path / "env"
    site = make_venv(root)
    shipped = site / "a.md"
    shipped.write_text("original", encoding="utf-8")

    # Stand in for uv's cache, and for a sibling environment.
    cached = tmp_path / "cache" / "a.md"
    cached.parent.mkdir()
    cached.hardlink_to(shipped)
    sibling = tmp_path / "sibling" / "a.md"
    sibling.parent.mkdir()
    sibling.hardlink_to(shipped)
    assert shipped.stat().st_nlink == 3

    source = tmp_path / "new.md"
    source.write_text("overlaid", encoding="utf-8")
    env = environs.Environment(
        pin=environs.Pin(name="a", version="1.0"), root=root
    )

    environs.apply_overlays(
        env, [environs.Overlay(source=source, destination="a.md")]
    )

    assert shipped.read_text(encoding="utf-8") == "overlaid"
    assert cached.read_text(encoding="utf-8") == "original"
    assert sibling.read_text(encoding="utf-8") == "original"
