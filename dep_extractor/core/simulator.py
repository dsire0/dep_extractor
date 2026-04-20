"""
dep_extractor/core/simulator.py
================================
Phase 3 — UV SAT-solver dry-run wrapper.

Educational: UV's `pip compile` acts as a SAT (Satisfiability) solver —
it finds the set of package versions that simultaneously satisfy all
requirements across all nodes, WITHOUT modifying your environment.

Key improvements over the monolith:
  - simulate_resolution() no longer calls sys.exit(1) — the CLI layer owns exits
  - parse_simulation_results() is now a private helper (it was never called externally)
  - parse_uv_causality() returns typed ConflictReport objects, not raw dicts
  - Python version selection is injected (AuditData.python_target_version) not hardcoded
  - All subprocess calls are wrapped with proper logging, no bare except: pass
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from dep_extractor.models import ConflictReport, SimulationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_resolution(
    req_files: list[Path],
    python_version: str = "3.12",
    extra_indices: list[str] | None = None,
    debug: bool = False,
) -> SimulationResult:
    """
    Runs `uv pip compile` in dry-run mode to validate the combined requirements.

    Args:
        req_files:      All requirements.txt files to compile together.
        python_version: Python version target for uv compile (e.g. '3.12', '3.13').
        extra_indices:  Additional PyTorch/hardware PyPI index URLs to include.
        debug:          If True, passes -v to uv for verbose output.

    Returns:
        SimulationResult with success flag, solved_map, and any parsed conflicts.

    Important: This function NEVER calls sys.exit(). If simulation fails,
    the caller (cli.py) is responsible for deciding the exit strategy.
    """
    if not req_files:
        logger.warning("simulate_resolution called with no req_files — returning empty success.")
        return SimulationResult(success=True)

    cmd = _build_compile_cmd(req_files, python_version, extra_indices, debug)

    if debug:
        files_display = [str(rf.parent.name) for rf in req_files[:5]]
        if len(req_files) > 5:
            files_display.append(f"... (+{len(req_files)-5} more)")
        logger.debug(
            "Running: uv pip compile on %d files: [%s] --python-version %s",
            len(req_files),
            ", ".join(files_display),
            python_version,
        )

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if process.returncode == 0:
            solved = _parse_simulation_output(process.stdout)
            return SimulationResult(
                success=True,
                solved_map=solved,
                raw_output=process.stdout,
            )
        else:
            conflicts = parse_uv_causality(process.stderr)
            return SimulationResult(
                success=False,
                raw_output=process.stderr,
                conflicts=conflicts,
            )

    except FileNotFoundError:
        logger.error("uv not found in PATH — cannot run SAT simulation.")
        return SimulationResult(
            success=False,
            raw_output="Error: uv not found in PATH.",
        )
    except Exception as exc:
        logger.error("Unexpected error during simulation: %s", exc)
        return SimulationResult(
            success=False,
            raw_output=str(exc),
        )


def parse_uv_causality(raw_error: str) -> list[ConflictReport]:
    """
    Deconstruct UV SAT-solver error output into structured ConflictReport objects.

    Educational: UV's error messages follow a tree format that states 'Because
    package A requires X>=1 and B requires X==0.9, we cannot satisfy all
    requirements.' This function extracts those key relationships.

    The monolith returned raw dicts; this returns typed ConflictReport objects
    so callers get IDE autocomplete and type checking.

    Args:
        raw_error: The full stderr output from a failed `uv pip compile` call.

    Returns:
        List of ConflictReport objects (may be empty if output doesn't match patterns).
    """
    conflicts: list[ConflictReport] = []
    seen: set[str] = set()

    # Pattern 1: "Because A depends on pkg>=X and B depends on pkg==Y"
    pattern1 = re.compile(
        r"Because (.*?) depends on ([a-zA-Z0-9_-]+)([~^=<>!]+[0-9a-zA-Z.*-]+)"
        r" and (.*?) depends on \2([~^=<>!]+[0-9a-zA-Z.*-]+)",
        re.IGNORECASE,
    )

    # Pattern 2: "pkg depends on X which conflicts with Y's dependencies"
    pattern2 = re.compile(
        r"(.*?) depends on ([a-zA-Z0-9_-]+)([~^=<>!]+[0-9a-zA-Z.*-]+)"
        r" which conflicts with (.*?)'s dependencies",
        re.IGNORECASE,
    )

    # Pattern 3: UV tree branch format "├─ pkg>=1.0" or "└─ pkg==2.0"
    pattern3 = re.compile(
        r"[├└]─\s+([a-zA-Z0-9_-]+)([~^=<>!]+[0-9a-zA-Z.*-]+)",
        re.IGNORECASE,
    )

    # Pattern 4: "No matching distribution found for XXX"
    pattern4 = re.compile(
        r"No matching distribution found for ([a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    )

    # Pattern 5: "Failed to build wheel for XXX"
    pattern5 = re.compile(
        r"Failed to build wheel for ([a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    )

    for pattern in (pattern1, pattern2, pattern3, pattern4, pattern5):
        for match in pattern.finditer(raw_error):
            groups = match.groups()
            if len(groups) >= 1:
                pkg = groups[1] if len(groups) > 1 else groups[0]
                spec = groups[2] if len(groups) > 2 else ""
                key = f"{pkg}{spec}"
                if key not in seen:
                    seen.add(key)
                    conflicts.append(
                        ConflictReport(
                            package=pkg.lower().replace("_", "-"),
                            specs=[spec] if spec else [],
                            nodes=[{"raw": match.group(0).strip()}],
                            severity="hard",
                        )
                    )

    return conflicts


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_compile_cmd(
    req_files: list[Path],
    python_version: str,
    extra_indices: list[str] | None,
    debug: bool,
) -> list[str]:
    """Constructs the uv pip compile command line."""
    cmd = ["uv", "pip", "compile"]
    cmd.extend(str(rf) for rf in req_files)
    cmd.extend(["--python-version", python_version, "--refresh"])

    # Only use --universal for modern stable Python (3.12+).
    # For older versions (e.g. 3.10), we resolve strictly for the target
    # environment to avoid conflict noise from 'other versions'.
    try:
        ver_major, ver_minor = map(int, python_version.split(".")[:2])
        if ver_major > 3 or (ver_major == 3 and ver_minor >= 12):
            cmd.append("--universal")
    except (ValueError, IndexError):
        # Fallback to absolute strict if version is weird
        pass

    if extra_indices:
        for idx in extra_indices:
            cmd.extend(["--extra-index-url", idx])

    if debug:
        cmd.append("-v")

    return cmd


def _parse_simulation_output(raw_output: str) -> dict[str, str]:
    """
    Parses `uv pip compile` stdout into a {package_name: version} map.

    UV compile output format:
        # via some-node
        numpy==1.26.4
        torch==2.3.1+cu121
        ...

    Educational: We only match lines that look like `name==version` (the
    pinned-resolution lines) and skip comments and option lines.

    Returns:
        dict mapping normalized package name → resolved version string.
    """
    solved: dict[str, str] = {}
    # Match lines like: 'numpy==1.26.4' or 'torch==2.3.1+cu121'
    pin_re = re.compile(r"^([a-zA-Z0-9_.-]+)==([^\s]+)")

    for line in raw_output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = pin_re.match(line)
        if match:
            name = match.group(1).lower().replace("_", "-")
            version = match.group(2)
            solved[name] = version

    return solved
