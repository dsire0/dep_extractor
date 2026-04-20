"""
dep_extractor/core/scanner.py
==============================
Phase 1 — Recursive node and requirements.txt discovery.

Educational: This module answers ONE question cleanly: "What requirement
files exist across all custom nodes, and what is in them?"

The original monolith did this inside `combine_requirements()`, which
also handled conflict detection, URL validation, and report writing.
Isolating discovery into a dedicated scanner makes it independently
testable — the test just creates a fake directory tree with tmp_path.

Key improvements over the monolith:
  - Uses Path.glob() with a clean skip predicate instead of os.walk loops
  - read_file_safe() is a proper importable utility, not a hidden closure
  - Returns a ScanResult dataclass instead of mutating 4 separate sets
  - Properly deduplicates packages by (name, specifier) across nodes
"""

from __future__ import annotations

import logging
from pathlib import Path

from dep_extractor.core.normalizer import RequirementNormalizer
from dep_extractor.models import NormalizedRequirement, ScanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

# Directories that are never scanned for requirements.txt files.
# These are either hidden dirs, version-control metadata, or
# intentionally-disabled nodes (StabilityMatrix convention).
_SKIP_DIR_PATTERNS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg",
    "__pycache__", ".mypy_cache", ".tox",
    ".disabled",
})


def _should_skip_dir(d: Path | str) -> bool:
    """
    Returns True if the directory should be excluded from scanning.

    Educational: Matching on the directory NAME (not full path) keeps this
    predicate fast and consistent regardless of nesting depth.
    """
    name = d.name if isinstance(d, Path) else d
    return name.startswith(".") or name in _SKIP_DIR_PATTERNS


def read_file_safe(path: Path) -> str:
    """
    Read a text file, trying multiple encodings in priority order.

    Encoding cascade:
        1. UTF-8 with BOM (utf-8-sig)  — most modern files
        2. UTF-16                       — rare but present in some Windows tools
        3. Latin-1 / ISO-8859-1        — guaranteed to succeed on any byte sequence

    Educational: The 'latin-1' codec never raises UnicodeDecodeError because
    every byte value maps to a valid character. It's a safe last resort.

    Returns:
        File content as a string. On permission/IO errors, returns "" and
        logs a warning (EAFP — don't crash the whole scan for one bad file).
    """
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (PermissionError, OSError) as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return ""
    return ""


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_nodes(
    root: Path,
    normalizer: RequirementNormalizer | None = None,
) -> ScanResult:
    """
    Walk `root` and collect all requirements from custom node subdirectories.

    Args:
        root:       The `custom_nodes` root directory.
        normalizer: Optional pre-configured normalizer. Constructed with
                    platform defaults if not provided.

    Returns:
        A fully populated ScanResult with all discovered requirements.

    Discovery rules:
        - Scans only IMMEDIATE subdirectories of `root` (one level deep).
          This matches the ComfyUI custom node layout.
        - Within each node dir, finds requirements.txt at any depth.
        - Skips hidden dirs, __pycache__, and .disabled nodes.
    """
    if normalizer is None:
        normalizer = RequirementNormalizer()

    result = ScanResult()
    seen_standard: dict[str, NormalizedRequirement] = {}  # name → first-seen req

    if not root.is_dir():
        logger.error("Scan root does not exist: %s", root)
        return result

    # Iterate immediate child directories only (custom_nodes/<node_name>/)
    for node_dir in sorted(root.iterdir()):
        if not node_dir.is_dir() or _should_skip_dir(node_dir):
            continue

        node_name = node_dir.name
        _scan_node_dir(node_dir, node_name, normalizer, result, seen_standard)

    logger.info(
        "Scan complete — %d nodes, %d req files, %d packages, %d specialized",
        len({r.source_node for r in result.packages}),
        len(result.req_files),
        len(result.packages),
        len(result.specialized),
    )
    return result


def _scan_node_dir(
    node_dir: Path,
    node_name: str,
    normalizer: RequirementNormalizer,
    result: ScanResult,
    seen_standard: dict,
) -> None:
    """
    Scans a single custom node directory for requirements.txt files.

    Mutates `result` in-place (performance: avoids rebuilding the list
    on every file). Since scan_nodes() is the sole caller, this is safe.
    """
    req_files = [
        p for p in node_dir.rglob("requirements.txt")
        if not any(_should_skip_dir(part) for part in p.parts)
    ]

    for req_file in sorted(req_files):
        result.req_files.append(req_file)
        _process_req_file(req_file, node_name, normalizer, result, seen_standard)


def _process_req_file(
    req_file: Path,
    node_name: str,
    normalizer: RequirementNormalizer,
    result: ScanResult,
    seen_standard: dict,
) -> None:
    """
    Reads a single requirements.txt and adds its contents to ScanResult.

    Deduplication strategy:
      - Standard packages: deduplicated by normalized name. If the same
        package appears in multiple nodes, it is kept once in `packages`
        but ALL origins are tracked in `origin_map` for conflict reporting.
      - Specialized (URL/path): deduplicated by raw line. Every unique
        URL is kept because git-pinned packages are opaque to pip/uv.
    """
    content = read_file_safe(req_file)
    if not content:
        return

    for line in content.splitlines():
        req = normalizer.normalize(line, node_name)
        if req is None:
            continue

        # Record the origin for conflict analysis
        origins = result.origin_map.setdefault(req.name, [])
        origins.append({"node": node_name, "raw": req.raw, "spec": req.specifier})

        if req.is_url:
            # Deduplicate by raw URL string
            if not any(s.raw == req.raw for s in result.specialized):
                result.specialized.append(req)
        else:
            # Deduplicate by normalized name; first occurrence wins
            if req.name not in seen_standard:
                seen_standard[req.name] = req
                result.packages.append(req)
            else:
                # Different specifier from a different node → potential conflict
                # (reported later by conflict_detector; only tracking origin here)
                pass
