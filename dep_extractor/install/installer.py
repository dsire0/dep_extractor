"""
dep_extractor/install/installer.py
=====================================
Phases 5 & 6 — Orchestrated installation manager.

Educational: This module is the "conductor" of the install pipeline. It does
NOT contain installation logic itself — it delegates:
  - Standard packages → single `uv pip install -r standard_requirements.txt`
  - Heavy packages → heavy_compiler.install_heavy_package()
  - Version checking → models._version_satisfies()

The original monolith had ~155 lines of deeply nested install logic within
`execute_phased_installation()`. Here the orchestrator is ~80 lines and relies
on the clearly-separated sub-modules for the actual work.

Key improvements:
  - get_pkg_name() is no longer an inner local function — imported from models
  - Version satisfaction uses the canonical _version_satisfies() from models.py
  - pip-fallback is isolated in heavy_compiler._try_install_with_fallback()
  - Returns a structured InstallSummary instead of a bare boolean
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

from dep_extractor.config import HEAVY_COMPILERS
from dep_extractor.install.heavy_compiler import install_heavy_package
from dep_extractor.install.toolchain import probe_toolchain
from dep_extractor.models import (
    AuditData,
    InstalledPackage,
    NormalizedRequirement,
    ScanResult,
    ToolchainStatus,
    _version_satisfies,
)
from dep_extractor.output.console import console, print_error, print_success, print_warning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class InstallSummary:
    """Structured result of the install phase."""
    base_installed: list[str] = field(default_factory=list)
    heavy_installed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_phased_installation(
    audit: AuditData,
    scan: ScanResult,
    installed: list[InstalledPackage],
) -> InstallSummary:
    """
    Orchestrate the full phased installation sequence.

    Phase 5 — Standard dependencies:
      Skips already-satisfied packages. Writes remaining to a temp
      requirements file and installs in one uv batch call.

    Phase 6 — Heavy compiler packages:
      Probes the toolchain once, then calls install_heavy_package()
      per package with the shared toolchain status.

    Args:
        audit:     Populated AuditData from run_hardware_audit().
        scan:      ScanResult containing all discovered requirements.
        installed: Current pip-installed package list.

    Returns:
        InstallSummary describing what happened.
    """
    summary = InstallSummary()

    # Build a fast lookup: normalized_name → installed version
    installed_map: dict[str, str] = {
        p.normalized_name: p.version for p in installed
    }
    installed_names: set[str] = set(installed_map.keys())

    all_reqs = scan.all_requirements

    # Split heavy vs standard
    heavy_reqs = [r for r in all_reqs if r.name in HEAVY_COMPILERS]
    standard_reqs = [r for r in all_reqs if r.name not in HEAVY_COMPILERS]

    # ------------------------------------------------------------------
    # Phase 5: Standard dependencies
    # ------------------------------------------------------------------
    console.print()
    console.print("[bold bright_blue]🚀 Phase 5: Standard Dependencies[/bold bright_blue]")

    to_install: list[NormalizedRequirement] = []

    for req in sorted(standard_reqs, key=lambda r: r.name):
        inst_ver = installed_map.get(req.name)
        is_installed = req.name in installed_names
        meets_spec = _version_satisfies(inst_ver, req.specifier) if is_installed else False

        # Check against simulation result if available
        solved_ver = audit.solved_map.get(req.name)
        matches_solved = (inst_ver == solved_ver) if (is_installed and solved_ver) else True

        if is_installed and meets_spec and matches_solved:
            summary.skipped.append(req.name)
        else:
            to_install.append(req)
            if is_installed:
                reason = "version mismatch" if not meets_spec else "differs from solved"
                console.print(
                    f"   [yellow]Mismatch[/yellow] {req.name}: "
                    f"installed={inst_ver} required={req.specifier or 'any'} "
                    f"solved={solved_ver or 'N/A'} → {reason}"
                )

    if summary.skipped:
        console.print(
            f"   [dim]Skipping {len(summary.skipped)} already-satisfied packages.[/dim]"
        )

    if to_install:
        console.print(f"   Installing [cyan]{len(to_install)}[/cyan] standard packages...")
        batch_ok = _batch_install(to_install)
        if batch_ok:
            summary.base_installed = [r.name for r in to_install]
            print_success(f"Standard installation complete.")
        else:
            summary.failed.extend(r.name for r in to_install)
            print_error("Standard installation failed.")
    else:
        print_success("All standard packages already satisfied.")

    # ------------------------------------------------------------------
    # Phase 6: Heavy compiler packages
    # ------------------------------------------------------------------
    if not heavy_reqs:
        return summary

    console.print()
    console.print(f"[bold bright_blue]☢️  Phase 6: Heavy Compiler Resolution ({len(heavy_reqs)} detected)[/bold bright_blue]")

    toolchain = probe_toolchain()

    for req in heavy_reqs:
        ok = install_heavy_package(req, toolchain, installed_names)
        if ok:
            summary.heavy_installed.append(req.name)
        else:
            summary.failed.append(req.name)

    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _batch_install(reqs: list[NormalizedRequirement]) -> bool:
    """
    Install a list of standard requirements in a single uv batch call.

    Writes requirements to a temp file and calls `uv pip install -r`.
    This is significantly faster than one pip call per package.

    Returns:
        True on success.
    """
    # Write to NamedTemporaryFile so we don't leave files behind on error
    try:
        content = "\n".join(str(r) for r in reqs)
        with NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="depextractor_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "-r", tmp_path],
            check=True,
        )
        return True

    except subprocess.CalledProcessError as exc:
        logger.error("Batch install failed (rc=%d)", exc.returncode)
        return False
    except Exception as exc:
        logger.error("Unexpected error during batch install: %s", exc)
        return False
    finally:
        # Clean up temp file even on failure
        try:
            Path(tmp_path).unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass
