"""
dep_extractor/analysis/conflict_detector.py
=============================================
Static version conflict analysis — the dep_extractor package's internal version.

Educational: This module performs PRE-RESOLUTION conflict detection.
It doesn't run the UV SAT solver; it simply checks whether any two nodes
declare incompatible versions of the same package (e.g., node A requires
numpy==1.24 while node B requires numpy==1.26).

This is useful because:
  1. It's instant (no subprocess call)
  2. It surfaces hard conflicts before wasting time on UV compilation
  3. It works even when UV is not installed

The EXTERNAL `conflict_detector.py` (in custom_nodes root) is a blessed
standalone tool. Its duplicate parser functions will be replaced with
imports from dep_extractor.core.normalizer and dep_extractor.models.
This module is the internal analysis engine that both tools share.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dep_extractor.core.normalizer import RequirementNormalizer
from dep_extractor.core.scanner import read_file_safe
from dep_extractor.models import ConflictReport, _specs_conflict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_static_conflicts(
    origin_map: dict[str, list[dict]],
) -> list[ConflictReport]:
    """
    Perform pre-resolution static conflict analysis against the origin map.

    The `origin_map` comes from ScanResult.origin_map and has the structure:
        { "numpy": [{"node": "node-a", "raw": "numpy==1.24", "spec": "==1.24"},
                    {"node": "node-b", "raw": "numpy>=1.26", "spec": ">=1.26"}] }

    Conflict classification:
      - "hard": Two nodes pin EXACT incompatible versions (==1.24 vs ==1.26)
      - "soft": Two nodes use range specs that may or may not conflict (heuristic)

    Args:
        origin_map: The ScanResult.origin_map dict.

    Returns:
        List of ConflictReport objects, sorted by severity (hard first).
    """
    conflicts: list[ConflictReport] = []

    for pkg_name, origins in origin_map.items():
        if len(origins) < 2:
            continue  # Only one node declares this package — no conflict possible

        # Collect all unique specifiers for this package
        all_specs = [o.get("spec") for o in origins if o.get("spec")]

        # Check all pairs for conflicts
        for i in range(len(all_specs)):
            for j in range(i + 1, len(all_specs)):
                spec_a = all_specs[i]
                spec_b = all_specs[j]

                if _specs_conflict(spec_a, spec_b):
                    conflicts.append(
                        ConflictReport(
                            package=pkg_name,
                            specs=[spec_a, spec_b],
                            nodes=origins,
                            severity="hard",
                        )
                    )
                    break  # One conflict per package is enough to flag it

    # Sort: hard conflicts first, then soft
    conflicts.sort(key=lambda c: (c.severity == "soft", c.package))

    if conflicts:
        logger.warning(
            "Static analysis found %d conflict(s): %s",
            len(conflicts),
            [c.package for c in conflicts],
        )
    else:
        logger.info("Static conflict analysis: no conflicts detected.")

    return conflicts


def map_requirements_to_nodes(root: Path) -> dict[str, list[dict]]:
    """
    Build the inverse index of requirements: package_name → [origin dicts].

    This function is a convenience entry point for callers that only have
    a root path (e.g., the standalone conflict_detector.py CLI). It performs
    a fresh scan internally.

    Args:
        root: The custom_nodes root directory.

    Returns:
        origin_map dict suitable for passing to detect_static_conflicts().
    """
    from dep_extractor.core.scanner import scan_nodes
    result = scan_nodes(root)
    return result.origin_map


def format_conflict_report_text(conflicts: list[ConflictReport]) -> str:
    """
    Format a list of ConflictReport objects as human-readable plain text.

    Used by the standalone conflict_detector.py when run in non-rich
    environments (e.g., piped output, CI logs, StabilityMatrix terminal).

    Returns:
        Multi-line string with one conflict per block.
    """
    if not conflicts:
        return "✅ No version conflicts detected.\n"

    lines: list[str] = [
        f"⚠️  Found {len(conflicts)} conflict(s):\n",
        "=" * 60,
    ]
    for c in conflicts:
        lines.append(f"\n[{c.severity.upper()}] {c.package}")
        lines.append(f"  Specifiers: {' vs '.join(c.specs)}")
        for node in c.nodes:
            lines.append(f"    - {node.get('node', '?')}: {node.get('raw', '?')}")

    return "\n".join(lines) + "\n"
