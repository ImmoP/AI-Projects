#!/usr/bin/env python3
"""Set up each project's isolated environment (if needed) and run its
smoke-marked tests, then print one summary.

This is the single command a reviewer should run to check that all five
projects' pipelines are still executable, straight after a fresh clone,
with nothing pre-installed: `python scripts/smoke.py` from the repo root.

For each project, this creates (or reuses) an isolated environment under
`.cache/smoke-envs/<project>/` (gitignored), installs that project's own
dependencies into it -- `uv` if available on PATH, otherwise `python -m
venv` + pip -- and then runs `pytest -m smoke` inside it. Nothing is
installed into the interpreter running this script itself. The first run
installs real dependencies (torch, transformers, ...) and can take several
minutes; use --no-install on later runs to skip straight to testing once
the environments already exist.

"Smoke" here means "does the pipeline run?", not "is the model good?" --
these are the fast, offline, no-network, no-real-model-call tests marked
`@pytest.mark.smoke` in each project's test suite. Environment setup does
use the network (to install packages); the tests themselves do not.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_CACHE_DIR = REPO_ROOT / ".cache" / "smoke-envs"

# "editable": the project has an installable pyproject.toml (a [project]
# table) with a "dev" extra -> `pip install -e ".[dev]"`.
# "requirements": no installable package, just a requirements.txt ->
# `pip install -r requirements.txt` (already includes pytest).
PROJECTS = {
    "tidy-agent": "editable",
    "food-finder": "requirements",
    "email-spam-detector": "editable",
    "rag-system": "requirements",
    "llm-from-scratch": "requirements",
}

# pytest exit code 5 = "no tests collected" -- treated as SKIPPED, not FAILED.
NO_TESTS_COLLECTED = 5


@dataclass
class ProjectResult:
    name: str
    status: str  # "PASS", "FAIL", "SKIPPED", "SETUP_FAILED"
    duration_seconds: float
    summary_line: str


def _last_meaningful_line(output: str) -> str:
    for line in reversed(output.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _venv_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _uv_available() -> bool:
    return shutil.which("uv") is not None


def _venv_module_works() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "venv", "--help"],
            capture_output=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def preflight_check(use_uv: bool) -> None:
    """Fail fast, once, with a single clear message -- not five identical
    per-project failures -- if there is no way to create an isolated
    environment at all."""
    if use_uv or _venv_module_works():
        return
    print(
        "ERROR: no way to create isolated environments.\n\n"
        "Neither `uv` is on PATH nor does `python -m venv` work with this "
        "interpreter (" + sys.executable + ").\n\n"
        "Fix one of the two, then re-run:\n\n"
        "  Install uv (recommended, https://docs.astral.sh/uv/):\n"
        "    curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux\n"
        "    powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"  # Windows\n\n"
        "  Or repair this Python's venv/ensurepip support, then:\n"
        "    python scripts/smoke.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _manual_setup_hint(name: str, install_mode: str) -> str:
    if install_mode == "editable":
        deps = 'pip install -e ".[dev]"'
    else:
        deps = "pip install -r requirements.txt"
    return (
        f"cd projects/{name} && python -m venv .venv && <activate .venv> && {deps}"
        f"  (see projects/{name}/README.md)"
    )


def ensure_env(name: str, install_mode: str, *, use_uv: bool) -> Path | None:
    """Create (if missing) and install into the project's isolated venv.
    Returns the venv's python executable, or None if setup failed."""
    project_dir = REPO_ROOT / "projects" / name
    env_dir = ENV_CACHE_DIR / name

    if not env_dir.exists():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        method = "uv" if use_uv else "python -m venv"
        print(f"==> {name}: creating environment ({method})...", flush=True)
        if use_uv:
            result = subprocess.run(["uv", "venv", str(env_dir), "--python", sys.executable])
        else:
            result = subprocess.run([sys.executable, "-m", "venv", str(env_dir)])
        if result.returncode != 0:
            return None

    venv_python = _venv_python(env_dir)
    deps_args = ["-e", ".[dev]"] if install_mode == "editable" else ["-r", "requirements.txt"]

    print(
        f"==> {name}: installing dependencies "
        f"(first run can take several minutes, e.g. torch/transformers)...",
        flush=True,
    )
    if use_uv:
        install_cmd = ["uv", "pip", "install", "--python", str(venv_python), *deps_args]
    else:
        install_cmd = [str(venv_python), "-m", "pip", "install", *deps_args]

    # Not captured: pip/uv's own progress output streams straight to the
    # terminal, so a multi-minute install doesn't look like a hang.
    result = subprocess.run(install_cmd, cwd=project_dir)
    if result.returncode != 0:
        return None

    return venv_python


def run_project(name: str, install_mode: str, *, use_uv: bool, no_install: bool) -> ProjectResult:
    project_dir = REPO_ROOT / "projects" / name
    env_dir = ENV_CACHE_DIR / name
    start = time.perf_counter()

    if no_install:
        venv_python = _venv_python(env_dir)
        if not venv_python.exists():
            duration = time.perf_counter() - start
            return ProjectResult(
                name, "SETUP_FAILED", duration,
                f"--no-install given but no environment at {env_dir} -- "
                "run once without --no-install first.",
            )
    else:
        venv_python = ensure_env(name, install_mode, use_uv=use_uv)
        if venv_python is None:
            duration = time.perf_counter() - start
            hint = _manual_setup_hint(name, install_mode)
            return ProjectResult(
                name, "SETUP_FAILED", duration,
                f"environment setup failed -- try manually: {hint}",
            )

    print(f"==> {name}: running smoke tests...", flush=True)

    # Fresh, per-run basetemp instead of pytest's shared default (a
    # `pytest-of-<user>` directory under the OS temp dir). That shared
    # directory can end up locked or permission-denied on Windows if a
    # previous run left it in a bad state -- a fresh one here sidesteps
    # that entirely and keeps the five projects' temp state isolated.
    with tempfile.TemporaryDirectory(prefix=f"smoke-{name}-") as basetemp:
        completed = subprocess.run(
            [str(venv_python), "-m", "pytest", "-m", "smoke", "-q", f"--basetemp={basetemp}"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

    duration = time.perf_counter() - start
    tail = _last_meaningful_line(completed.stdout) or _last_meaningful_line(completed.stderr)

    if completed.returncode == 0:
        status = "PASS"
    elif completed.returncode == NO_TESTS_COLLECTED:
        status = "SKIPPED"
    else:
        status = "FAIL"

    return ProjectResult(name=name, status=status, duration_seconds=duration, summary_line=tail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "projects",
        nargs="*",
        help=f"Subset of projects to run (default: all). Choices: {', '.join(PROJECTS)}.",
    )
    parser.add_argument(
        "--no-install", "--fast",
        dest="no_install",
        action="store_true",
        help="Skip environment setup entirely; reuse the cached venvs as-is. "
             "Fails per-project with SETUP_FAILED if a venv doesn't exist yet.",
    )
    args = parser.parse_args()

    selected = args.projects or list(PROJECTS)
    unknown = [name for name in selected if name not in PROJECTS]
    if unknown:
        parser.error(f"unknown project(s): {', '.join(unknown)} (choices: {', '.join(PROJECTS)})")

    use_uv = _uv_available()
    if not args.no_install:
        preflight_check(use_uv)

    env_manager = "uv" if use_uv else "python -m venv + pip (uv not found on PATH)"
    print(f"Running smoke tests for: {', '.join(selected)}")
    print(f"Environment manager: {env_manager}")
    print(f"Environment cache: {ENV_CACHE_DIR}\n")

    results = [
        run_project(name, PROJECTS[name], use_uv=use_uv, no_install=args.no_install)
        for name in selected
    ]

    print()
    name_width = max(len(r.name) for r in results)
    print(f"{'PROJECT':<{name_width}}  STATUS         TIME   RESULT")
    print("-" * (name_width + 55))
    for r in results:
        line = f"{r.name:<{name_width}}  {r.status:<13}  {r.duration_seconds:5.1f}s  "
        print(line + r.summary_line)

    failed = [r for r in results if r.status == "FAIL"]
    setup_failed = [r for r in results if r.status == "SETUP_FAILED"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    passed_count = len(results) - len(failed) - len(setup_failed) - len(skipped)

    print()
    print(
        f"{passed_count} passed, {len(failed)} failed, "
        f"{len(setup_failed)} setup failed, {len(skipped)} skipped"
    )

    if failed:
        print(f"Test failures: {', '.join(r.name for r in failed)}")
    if setup_failed:
        print(f"Environment setup failures: {', '.join(r.name for r in setup_failed)}")

    return 1 if (failed or setup_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
