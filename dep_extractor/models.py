"""
dep_extractor/models.py
=======================
Shared data models for the ComfyUI Dependency Resolver Suite.

All data flowing between pipeline stages is represented as typed @dataclass
objects. This eliminates the raw `dict` soup used by the original monolith
and makes every stage independently testable and type-checkable.

Educational: Using dataclasses over plain dicts provides:
  - IDE autocomplete on every field
  - Runtime TypeError on missing required fields
  - Clean __repr__ for free (great for debug logging)
  - Zero overhead vs a hand-written class
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Requirement Models
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRequirement:
    """
    A fully parsed and normalized dependency requirement.

    Produced by RequirementNormalizer.normalize() for every non-comment,
    non-blank line in a node's requirements.txt.

    Field notes:
      - `name`       : lowercase, hyphen-normalized (e.g. 'onnx-runtime')
      - `specifier`  : version constraint string (e.g. '>=1.2.0,<2.0')
      - `is_url`     : True for git+/http/local-path dependencies
      - `is_heavy`   : True if this package requires MSVC/CUDA compilation
      - `source_node`: The custom_node folder name this requirement came from
    """
    raw: str
    name: str
    specifier: str
    is_url: bool
    is_heavy: bool
    source_node: str

    def __str__(self) -> str:
        """Reconstructs the installable requirement string."""
        if self.is_url:
            return self.raw
        if self.specifier:
            return f"{self.name}{self.specifier}"
        return self.name

    def __hash__(self) -> int:
        # Deduplicate by (name, specifier) — not by raw string
        return hash((self.name, self.specifier, self.is_url))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizedRequirement):
            return NotImplemented
        return self.name == other.name and self.specifier == other.specifier and self.is_url == other.is_url


@dataclass
class InstalledPackage:
    """A single package entry from `pip list --format=json`."""
    name: str
    version: str

    @property
    def normalized_name(self) -> str:
        """Returns the pip-canonical normalized name (lower, hyphenated)."""
        return self.name.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Scan Result
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    The complete output of Phase 1 (node scanning).

    Holds every requirement discovered across all custom nodes,
    split into standard package requirements and specialized
    URL/path/wheel dependencies.
    """
    req_files: list = field(default_factory=list)          # list[Path]
    packages: list = field(default_factory=list)           # list[NormalizedRequirement] — standard pkgs
    specialized: list = field(default_factory=list)        # list[NormalizedRequirement] — URLs/paths
    origin_map: dict = field(default_factory=dict)         # pkg_name → [{"node": str, "raw": str}]

    @property
    def online_urls(self) -> list[str]:
        """Returns only http/https/git+ URLs from specialized requirements."""
        return [
            r.raw for r in self.specialized
            if r.raw.startswith(("http://", "https://", "git+http"))
        ]

    @property
    def all_requirements(self) -> list:
        """Combined list of all requirements (standard + specialized)."""
        return self.packages + self.specialized


# ---------------------------------------------------------------------------
# Hardware Audit
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    """Information about a single detected GPU."""
    gpu_type: str       # "NVIDIA", "Other"
    name: str
    vram_gb: int


@dataclass
class AuditData:
    """
    Full GINGER-standard hardware and environment profile.

    Produced by run_hardware_audit(). Used to steer the UV SAT-solver
    toward correct platform-specific dependencies and CUDA indices.

    Field notes:
      - `is_ginger_class`: True if a 4090 or 5090 GPU is detected
      - `is_insider`     : True if Windows build > 22621 (Insider channel)
      - `solved_map`     : Populated only after simulate_resolution() succeeds
    """
    os: str = field(default_factory=platform.system)
    os_release: str = field(default_factory=platform.release)
    os_version: str = field(default_factory=platform.version)
    cpu: str = field(default_factory=platform.processor)
    is_insider: bool = False
    is_ginger_class: bool = False
    gpus: list = field(default_factory=list)               # list[GpuInfo]
    total_vram_gb: int = 0
    cuda_version: str | None = None
    has_uv: bool = False
    uv_path: str | None = None
    terminal: str = "Unknown"
    solved_map: dict = field(default_factory=dict)         # pkg_name → solved_version

    @property
    def python_target_version(self) -> str:
        """
        Educational: GINGER Insider builds support Python 3.13 targets.
        Non-insider environments default to the stable 3.12 target.
        """
        return "3.13" if self.is_insider else "3.12"


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------

@dataclass
class ConflictReport:
    """
    A detected version conflict between two or more custom nodes.

    Severity levels:
      - "hard" : Two nodes pin incompatible exact versions (==1.0 vs ==2.0)
      - "soft" : Two nodes use overlapping range specs — may or may not resolve
    """
    package: str
    specs: list[str]
    nodes: list[dict]
    severity: Literal["hard", "soft"] = "soft"

    def __str__(self) -> str:
        node_names = ", ".join(n.get("node", "?") for n in self.nodes)
        return (
            f"[{self.severity.upper()}] {self.package}: "
            f"{' vs '.join(self.specs)} "
            f"(from: {node_names})"
        )


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

@dataclass
class UrlValidationResult:
    """Result of validating a single online dependency URL."""
    url: str
    is_valid: bool
    error: str | None = None
    latency_ms: float = 0.0
    method_used: str = "HEAD"      # "HEAD" or "GET" (fallback)

    @property
    def status_label(self) -> str:
        return "VALID" if self.is_valid else f"INVALID: {self.error}"


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Output of the UV SAT-solver dry-run (uv pip compile).

    Educational: A SAT (Satisfiability) solver checks if there exists
    a mathematically valid set of package versions satisfying all
    constraints simultaneously — without actually installing anything.
    """
    success: bool
    solved_map: dict = field(default_factory=dict)         # pkg_name → solved_version
    raw_output: str = ""
    conflicts: list = field(default_factory=list)          # list[ConflictReport]


# ---------------------------------------------------------------------------
# Audit Report (aggregated for report writer)
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """
    Aggregated diagnostic data written to the audit section of
    combined_requirements.txt. Purely a data carrier for report_writer.py.
    """
    untracked_wheels: list[str] = field(default_factory=list)
    missing_from_reqs: list[str] = field(default_factory=list)
    version_mismatches: list[str] = field(default_factory=list)
    conflict_causality: list = field(default_factory=list)   # list[ConflictReport]


# ---------------------------------------------------------------------------
# Toolchain Status
# ---------------------------------------------------------------------------

@dataclass
class ToolchainStatus:
    """
    Result of probing the local build toolchain.

    Used by the install layer to determine whether heavy compiler
    packages (deepspeed, flash-attn, xformers) can be compiled natively.
    """
    has_msvc: bool = False
    vs_install_path: str | None = None
    has_ninja: bool = False
    has_cmake: bool = False
    has_cuda_toolkit: bool = False
    vcvars_env: dict = field(default_factory=dict)   # Injected MSVC build env

    @property
    def is_ready_for_heavy_build(self) -> bool:
        """True only when MSVC and Ninja are both present."""
        return self.has_msvc and self.has_ninja


# ---------------------------------------------------------------------------
# Version satisfaction helpers
# ---------------------------------------------------------------------------

def _version_satisfies(version: str | None, spec: str | None) -> bool:
    """
    Checks whether `version` satisfies the given version `spec` string.

    This is the single canonical implementation — it replaces the
    duplicated helpers that existed in both requirements_extractor.py
    and conflict_detector.py.

    Prefers the `packaging` library for accurate PEP 440 semantics.
    Falls back to a simple string comparison for the == operator only.
    """
    if not version or not spec:
        return True

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(version) in SpecifierSet(spec)
    except Exception:
        pass

    # Simple fallback: exact match only
    if spec.startswith("=="):
        return version == spec[2:]
    return True


def _specs_conflict(spec1: str | None, spec2: str | None) -> bool:
    """
    Returns True if two version specifier strings are mutually exclusive.

    Educational: Two specs conflict when no version can satisfy both
    simultaneously (e.g. ==1.0 and ==2.0 are always in conflict).
    """
    if not spec1 or not spec2:
        return False
    if spec1 == spec2:
        return False
    # Hard conflict: both are exact pins to different versions
    if spec1.startswith("==") and spec2.startswith("=="):
        return spec1 != spec2
    # Range conflict analysis would go here for production hardening
    return False
