"""The simulator registry, and the boundary it exists to keep.

The registry's whole purpose is that generic code never names a simulator, so the interesting
test here is not `register` round-tripping: it is the grep guard at the bottom, which fails the
moment a simulator name leaks back into the core.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from icilval import simulators
from icilval.simulators import Simulator


def _sim(name: str, **kw) -> Simulator:
    base = dict(
        run_units=lambda *a, **k: None,
        build_stage=lambda *a, **k: None,
        make_unit=lambda *a, **k: None,
        demo_frames=lambda demo: [],
        validate_skill=lambda skill, doc: [],
    )
    base.update(kw)
    return Simulator(name=name, **base)


#: The simulators this repository ships. A benchmark from another repository may also be
#: installed, which is the point of the registry - so these are a subset, never the whole list.
IN_REPO = {"libero", "draw"}


def test_the_shipped_simulators_are_registered():
    assert IN_REPO <= set(simulators.names())
    assert simulators.get("libero").name == "libero"
    assert all(simulators.get(name).distribution == "icilval" for name in IN_REPO)


def test_registering_a_name_twice_is_refused(monkeypatch):
    monkeypatch.setitem(simulators.REGISTRY, "fake", _sim("fake"))
    with pytest.raises(ValueError, match="registered twice"):
        simulators.register(_sim("fake"))


def test_an_unknown_simulator_names_the_ones_there_are():
    with pytest.raises(KeyError) as exc:
        simulators.get("nope")
    message = str(exc.value)
    assert "nope" in message and "libero" in message


def test_for_skill_reads_the_skill_s_simulator(spec):
    for skill in spec.skills("sensorimotor"):
        assert simulators.for_skill(spec, skill).name == spec.simulator(skill)


def test_an_uninstalled_benchmark_names_what_is_registered(spec):
    """A field may name a benchmark from another repository. Until that is installed, asking for
    it must say what there is rather than fail obscurely - and this must hold whether or not one
    happens to be pip-installed in the environment running the suite."""
    with pytest.raises(KeyError) as exc:
        simulators.get("nosuchsim")
    assert "nosuchsim" in str(exc.value) and "libero" in str(exc.value)


def test_a_simulator_need_not_check_its_pool_tasks():
    """The hook is optional: a simulator with no file format of its own supplies nothing."""
    assert _sim("bare").verify_pool_task(None, None) == []


def test_libero_refuses_an_unparsable_bddl(tmp_path):
    (tmp_path / "broken.bddl").write_text("(define (problem")

    class _Pool:
        def path(self, rel):
            return tmp_path / rel

    class _Task:
        bddl = "broken.bddl"

    errors = simulators.get("libero").verify_pool_task(_Pool(), _Task())
    assert errors and "bddl unparsable" in errors[0]


# ------------------------------------------------------------------ the boundary

#: Names a simulator owns. Outside `simulators/`, none of them may appear in the core: the whole
#: point of the registry is that generic code dispatches on `skills.<skill>.simulator` instead.
#: `robotwin`, `sapien` and `uniskill` are listed before they exist so the boundary cannot rot
#: when the first out-of-repo benchmark arrives.
SIMULATOR_WORDS = ("libero", "draw", "robotwin", "sapien", "uniskill")


#: `draw` is also an ordinary English verb and a matplotlib-ish method name, so the guard looks
#: for the quoted string literal, which is how a dispatch would actually be written.
def _quoted(word: str) -> tuple[str, ...]:
    return (f'"{word}"', f"'{word}'")


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "spec.json").exists():
            return parent
    return None


def test_no_module_outside_simulators_names_a_simulator():
    root = _repo_root()
    if root is None:  # installed as a wheel, with no source tree to walk
        pytest.skip("not a source checkout")
    core = root / "src" / "icilval"
    offenders: list[str] = []
    for path in sorted(core.rglob("*.py")):
        rel = path.relative_to(core)
        if rel.parts[0] == "simulators":
            continue
        text = path.read_text()
        for word in SIMULATOR_WORDS:
            if any(q in text for q in _quoted(word)):
                offenders.append(f"{rel}: {word}")
    assert not offenders, (
        "a simulator is named outside src/icilval/simulators/; dispatch through the registry "
        f"instead: {offenders}"
    )


def _packages() -> set[str]:
    root = _repo_root()
    if root is None:
        return set()
    pkg = root / "src" / "icilval" / "simulators"
    return {p.name for p in pkg.iterdir() if p.is_dir() and (p / "__init__.py").exists()}


def test_every_simulator_package_registers_itself():
    """`simulators/__init__` imports each package for its `register()` side effect. A package
    added without that import would be dead code, and the spec would reject its skills."""
    if _repo_root() is None:
        pytest.skip("not a source checkout")
    assert _packages() <= set(simulators.names())
    assert _packages() == IN_REPO


#: The libraries a simulator package must not pull in at module scope. `icilval spec validate`,
#: the queue and the dashboard's contract checks all import `icilval.simulators`, and none of
#: them has LIBERO, torch or a display.
HEAVY = ("libero", "robosuite", "torch", "pygame", "pymunk", "behavior_prompting", "h5py", "zarr")


def test_a_simulator_package_imports_nothing_heavy_at_module_scope():
    """Simulator imports are function-local, so the host CLI imports the registry without them."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    offenders: list[str] = []
    for name in sorted(_packages()):
        path = root / "src" / "icilval" / "simulators" / name / "__init__.py"
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in HEAVY:
                    offenders.append(f"simulators/{name}/__init__.py: {module}")
    assert not offenders, f"import these inside the function that needs them: {offenders}"
