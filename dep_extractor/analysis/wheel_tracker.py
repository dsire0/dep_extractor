"""
dep_extractor/analysis/wheel_tracker.py
=========================================
Discovers and classifies local .whl files in the workspace.

Educational: Some custom nodes bundle pre-built wheel files for heavy
dependencies (deepspeed, flash-attn, etc.) that are difficult to compile
from source on Windows. This module finds all such wheels and cross-
references them against the specialized requirements list to identify:

  - "Tracked" wheels: a node's requirements.txt already references this .whl
  - "Untracked" wheels: a .whl file exists but no requirements.txt covers it

Untracked wheels are worth investigating — they may be leftover artifacts
or manual installs that should be formally declared in a node's requirements.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dep_extractor.models import NormalizedRequirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_workspace_wheels(root: Path) -> list[Path]:
    """
    Recursively discover all .whl files in the workspace.

    Excludes:
      - Hidden directories (starting with '.')
      - Directories named '.disabled' (StabilityMatrix convention)
      - The dep_extractor package directory itself

    Args:
        root: The custom_nodes root directory.

    Returns:
        Sorted list of discovered .whl file paths.
    """
    if not root.is_dir():
        logger.warning("Wheel search root does not exist: %s", root)
        return []

    wheels = [
        p for p in root.rglob("*.whl")
        if not _is_excluded_path(p)
    ]
    logger.info("Found %d .whl files in workspace.", len(wheels))
    return sorted(wheels)


def classify_wheels(
    wheels: list[Path],
    specialized: list[NormalizedRequirement],
    root: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Split discovered wheels into tracked and untracked.

    A wheel is "tracked" if any entry in `specialized` (URL/path requirements)
    references its filename. An exact filename match is used — this is the
    reliable key since wheel filenames encode name, version, platform, and ABI.

    Args:
        wheels:      Output of find_workspace_wheels().
        specialized: Specialized requirements (URL/path deps) from ScanResult.
        root:        Root path for computing relative paths in log output.

    Returns:
        (tracked_wheels, untracked_wheels)
    """
    # Build a set of referenced wheel filenames from specialized requirements
    referenced_filenames: set[str] = set()
    for req in specialized:
        raw = req.raw
        # Local path reference: "/path/to/package-1.0-py3-none-any.whl"
        if raw.endswith(".whl"):
            referenced_filenames.add(Path(raw).name)

    tracked: list[Path] = []
    untracked: list[Path] = []

    for wheel in wheels:
        if wheel.name in referenced_filenames:
            tracked.append(wheel)
        else:
            untracked.append(wheel)
            logger.debug(
                "Untracked wheel detected: %s",
                wheel.relative_to(root) if root in wheel.parents else wheel,
            )

    logger.info(
        "Wheel classification: %d tracked, %d untracked",
        len(tracked),
        len(untracked),
    )
    return tracked, untracked


def get_wheel_package_name(wheel_path: Path) -> str:
    """
    Extract the normalized package name from a wheel filename.

    Educational: Wheel filename format (PEP 427):
        {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl

    The distribution (package name) is the first segment, with hyphens
    and underscores both valid. We normalize to hyphens (pip canonical form).

    Examples:
        flash_attn-2.5.0-cp310-cp310-win_amd64.whl → 'flash-attn'
        deepspeed-0.14.0-py3-none-any.whl → 'deepspeed'
    """
    stem = wheel_path.stem  # strips .whl suffix
    name_part = stem.split("-")[0]
    return name_part.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_excluded_path(path: Path) -> bool:
    """
    Returns True if any path component should cause this path to be skipped.

    Excluded: hidden dirs (dot-prefix), .disabled nodes, and __pycache__.
    """
    excluded_names = {"__pycache__", ".disabled", "dep_extractor"}
    for part in path.parts:
        if part.startswith(".") or part in excluded_names:
            return True
    return False
