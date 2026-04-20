"""
dep_extractor/core/normalizer.py
=================================
Unified requirement line parser and normalizer.

Educational: The original monolith contained TWO near-identical parse
functions — `normalize_requirement()` and `parse_requirement_line()` —
with ~60% code overlap. This single class replaces both.

Design: The `RequirementNormalizer` is a stateful class because it
requires the current OS to apply platform remaps at parse time. Making
OS-detection a constructor argument makes this class trivially testable
by injecting "Windows" or "Linux" without mocking platform.system().
"""

from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports

from dep_extractor.config import HEAVY_COMPILERS, PLATFORM_REMAPS
from dep_extractor.models import NormalizedRequirement


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a PEP 508 package name optionally followed by version specifiers.
# Examples:
#   torch>=2.0.0,<3.0
#   onnxruntime-gpu==1.16.0
#   requests
_NAME_SPEC_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"   # package name
    r"([!=<>~^].*)?"                                     # optional specifier
)

# Matches URL-style requirement prefixes (git, http, local wheel)
_URL_PREFIXES = (
    "git+", "http://", "https://", "file://",
    "./", "../", "/",
)

# Lines to silently skip (pip options, blank, comments)
_SKIP_PREFIXES = ("-r ", "-c ", "-e ", "#", "--")


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class RequirementNormalizer:
    """
    Parses and normalizes a raw requirements.txt line into a
    `NormalizedRequirement` dataclass.

    Usage:
        normalizer = RequirementNormalizer()
        req = normalizer.normalize("torch>=2.0", "comfyui-manager")
        # NormalizedRequirement(name='torch', specifier='>=2.0', ...)

    Thread safety: This class is stateless after __init__; it is safe to
    share a single instance across threads.
    """

    def __init__(
        self,
        current_os: str | None = None,
        platform_remaps: dict | None = None,
    ) -> None:
        """
        Args:
            current_os: OS name string (e.g. 'Windows'). Defaults to
                platform.system() when None.
            platform_remaps: Override the default PLATFORM_REMAPS dict.
                Useful for testing remaps without modifying global state.
        """
        self._os = current_os or platform.system()
        self._remaps: dict[str, str] = (
            platform_remaps if platform_remaps is not None
            else PLATFORM_REMAPS.get(self._os, {})
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        raw_line: str,
        source_node: str,
    ) -> NormalizedRequirement | None:
        """
        Parse a single raw requirements.txt line.

        Returns:
            NormalizedRequirement if the line represents an installable
            dependency, None for comments, blank lines, and pip option flags.

        Examples of lines that return None:
            ""                  # blank
            "# a comment"       # comment
            "--index-url ..."   # pip option
            "-r other.txt"      # file include directive
        """
        line = raw_line.strip()

        # Skip non-dependency lines
        if not line or any(line.startswith(p) for p in _SKIP_PREFIXES):
            return None

        # Inline comment stripping: 'torch>=2.0  # needed for ComfyUI'
        if " #" in line:
            line = line[: line.index(" #")].strip()

        # Editable installs: -e path/to/package  or -e git+...
        if line.startswith("-e "):
            line = line[3:].strip()

        # URL-style requirements (git+, http://, local paths)
        if self._is_url(line):
            name = self._extract_egg_name(line) or self._url_to_slug(line)
            name = self._normalize_name(name)
            return NormalizedRequirement(
                raw=raw_line.strip(),
                name=name,
                specifier="",
                is_url=True,
                is_heavy=name in HEAVY_COMPILERS,
                source_node=source_node,
            )

        # Standard name[extras]>=specifier format
        return self._parse_standard(raw_line.strip(), line, source_node)

    def normalize_name_only(self, raw_name: str) -> str:
        """
        Normalizes a package name string to its pip canonical form.

        Canonical form: lowercase, underscores replaced with hyphens.
        This is the PEP 503 "normalized" name used by PyPI.

        Educational: 'Torch', 'torch', 'TORCH' all refer to the same
        PyPI package. pip/uv normalise on the fly; we must too.
        """
        return raw_name.lower().replace("_", "-").replace(".", "-")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_standard(
        self,
        raw: str,
        cleaned: str,
        source_node: str,
    ) -> NormalizedRequirement | None:
        """Handles 'pkg[extra]>=spec' style lines."""
        # Strip extras notation: package[extra1,extra2]>=1.0 → package>=1.0
        no_extras = re.sub(r"\[.*?\]", "", cleaned)

        match = _NAME_SPEC_RE.match(no_extras)
        if not match:
            # Malformed line; skip silently (EAFP: log at debug level if needed)
            return None

        raw_name = match.group(1)
        raw_spec = (match.group(3) or "").strip()

        name = self._normalize_name(raw_name)
        name = self._apply_platform_remap(name)

        return NormalizedRequirement(
            raw=raw,
            name=name,
            specifier=raw_spec,
            is_url=False,
            is_heavy=name in HEAVY_COMPILERS,
            source_node=source_node,
        )

    def _normalize_name(self, name: str) -> str:
        """Applies pip canonical normalization to a package name."""
        return name.lower().replace("_", "-").replace(".", "-")

    def _apply_platform_remap(self, name: str) -> str:
        """
        Substitutes platform-specific package names.

        Example: 'triton' → 'triton-windows' on Windows systems.
        This ensures the correct variant is used in the generated
        combined_requirements.txt without manual editing.
        """
        return self._remaps.get(name, name)

    @staticmethod
    def _is_url(line: str) -> bool:
        """True if the line looks like a URL or local path dependency."""
        return any(line.startswith(prefix) for prefix in _URL_PREFIXES)

    @staticmethod
    def _extract_egg_name(url: str) -> str | None:
        """
        Extracts the #egg=name fragment from a URL requirement.

        Example: 'git+https://github.com/org/repo.git#egg=my-package'
                  → 'my-package'
        """
        if "#egg=" in url:
            return url.split("#egg=", 1)[1].split("&")[0].strip()
        return None

    @staticmethod
    def _url_to_slug(url: str) -> str:
        """
        Derives a human-readable package slug from a URL when no
        egg fragment is present. Strips scheme, host, .git suffix.

        Example: 'git+https://github.com/Dao-AILab/flash-attention.git'
                  → 'flash-attention'
        """
        # Strip git+ prefix if present
        clean = url.replace("git+", "").split("@")[0].split("?")[0]
        # Take the last path component, strip .git
        slug = clean.rstrip("/").split("/")[-1]
        if slug.endswith(".git"):
            slug = slug[:-4]
        return slug or "unknown-url-dep"
