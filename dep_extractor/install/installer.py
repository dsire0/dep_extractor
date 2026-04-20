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

from dep_extractor.config import HEAVY_COMPILERS, TORCH_SUITE_PREFIXES
from dep_extractor.core.auditor import suggest_best_indices
from dep_extractor.install.heavy_compiler import get_build_env, install_heavy_package
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
    extra_indices: list[str] | None = None,
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

    # Probe toolchain once for all remaining phases
    toolchain = probe_toolchain()

    all_reqs = scan.all_requirements

    # ------------------------------------------------------------------
    # Partitioning Logic
    # ------------------------------------------------------------------
    # Educational: The "Torch Suite" (torch, xformers, etc.) must be handled
    # as a unified batch with a locked index to prevent CUDA version drift.
    # This takes priority over the general "heavy" or "standard" classification.
    
    torch_reqs = [
        r for r in all_reqs
        if any(r.name.startswith(p) for p in TORCH_SUITE_PREFIXES)
    ]
    
    remaining_reqs = [r for r in all_reqs if r not in torch_reqs]
    
    heavy_reqs = [r for r in remaining_reqs if r.name in HEAVY_COMPILERS]
    other_reqs = [r for r in remaining_reqs if r.name not in HEAVY_COMPILERS]

    # ------------------------------------------------------------------
    # Phase 5: Non-Torch Standard Dependencies
    # ------------------------------------------------------------------
    console.print()
    console.print("[bold bright_blue]🚀 Phase 5: Standard Dependencies (Non-Torch)[/bold bright_blue]")

    to_install_others = _filter_satisfied(other_reqs, installed_map, audit)

    if to_install_others:
        console.print(f"   Installing [cyan]{len(to_install_others)}[/cyan] packages...")
        batch_ok = _batch_install(to_install_others, audit.python_executable, extra_indices, audit.solved_map)
        if batch_ok:
            summary.base_installed.extend(r.name for r in to_install_others)
            print_success(f"Standard installation (batch) complete.")
        else:
            summary.failed.extend(r.name for r in to_install_others)
            print_error("Standard batch installation failed.")
    else:
        print_success("All non-torch standard packages already satisfied.")

    # ------------------------------------------------------------------
    # Phase 5.5: Torch Suite Dependencies (CUDA Consistency Lock)
    # ------------------------------------------------------------------
    to_install_torch = _filter_satisfied(torch_reqs, installed_map, audit)

    if to_install_torch:
        console.print()
        console.print("[bold bright_blue]🔥 Phase 5.5: Torch Suite (CUDA Isolation)[/bold bright_blue]")
        console.print("   [dim]Locking a single index for all torch-related packages to prevent CUDA mismatch.[/dim]")

        # Get suggested indices from hardware audit
        indices = suggest_best_indices(audit)
        if not indices:
            print_warning("No hardware-specific indices suggested. Falling back to default PyPI.")
            indices = [None]  # type: ignore

        torch_ok = _torch_sequential_install(
            to_install_torch,
            audit.python_executable,
            indices,
            toolchain,
            audit.solved_map
        )

        if torch_ok:
            summary.base_installed.extend(r.name for r in to_install_torch)
            print_success("Torch Suite installation complete.")
        else:
            summary.failed.extend(r.name for r in to_install_torch)
            print_error("Torch Suite installation failed or was skipped.")
    else:
        if torch_reqs:
            print_success("All Torch Suite packages already satisfied.")

    # ------------------------------------------------------------------
    # Phase 6: Heavy compiler packages
    # ------------------------------------------------------------------
    if not heavy_reqs:
        return summary

    console.print()
    console.print(f"[bold bright_blue]☢️  Phase 6: Heavy Compiler Resolution ({len(heavy_reqs)} detected)[/bold bright_blue]")

    skip_all_heavy = False
    for req in heavy_reqs:
        if skip_all_heavy:
            summary.failed.append(req.name)
            continue

        ok = install_heavy_package(req, toolchain, installed_names, audit)
        if ok is True:
            summary.heavy_installed.append(req.name)
        elif ok is None:
            skip_all_heavy = True
            summary.failed.append(req.name)
        else:
            summary.failed.append(req.name)

    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _filter_satisfied(
    reqs: list[NormalizedRequirement],
    installed_map: dict[str, str],
    audit: AuditData,
) -> list[NormalizedRequirement]:
    """Return a list of requirements that are NOT already satisfied."""
    to_install = []
    for req in sorted(reqs, key=lambda r: r.name):
        inst_ver = installed_map.get(req.name)
        is_installed = req.name in installed_map
        meets_spec = _version_satisfies(inst_ver, req.specifier) if is_installed else False

        # Check against simulation result
        solved_ver = audit.solved_map.get(req.name)
        matches_solved = (inst_ver == solved_ver) if (is_installed and solved_ver) else True

        if not (is_installed and meets_spec and matches_solved):
            to_install.append(req)
            if is_installed:
                reason = "version mismatch" if not meets_spec else "differs from solved"
                console.print(
                    f"   [yellow]Mismatch[/yellow] {req.name}: "
                    f"installed={inst_ver} required={req.specifier or 'any'} "
                    f"solved={solved_ver or 'N/A'} → {reason}"
                )
    return to_install


def _batch_install(
    reqs: list[NormalizedRequirement],
    python_executable: str,
    extra_indices: list[str] | None = None,
    solved_map: dict[str, str] | None = None,
) -> bool:
    """
    Install a list of standard requirements in a single uv batch call.
    """
    # Write to NamedTemporaryFile so we don't leave files behind on error
    tmp_path = None
    try:
        lines = []
        for r in reqs:
            solved_ver = (solved_map or {}).get(r.name)
            if solved_ver and not r.is_url:
                lines.append(f"{r.name}=={solved_ver}")
            else:
                lines.append(str(r))

        content = "\n".join(lines)
        with NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="depextractor_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cmd = ["uv", "pip", "install", "--python", python_executable, "-r", tmp_path]
        if extra_indices:
            for idx in extra_indices:
                cmd.extend(["--extra-index-url", idx])
            cmd.extend(["--index-strategy", "unsafe-best-match"])

        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as exc:
        logger.error("Batch install failed (rc=%d)", exc.returncode)
        return False
    except Exception as exc:
        logger.error("Unexpected error during batch install: %s", exc)
        return False
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _torch_sequential_install(
    reqs: list[NormalizedRequirement],
    python_executable: str,
    suggested_indices: list[str],
    toolchain: ToolchainStatus,
    solved_map: dict[str, str] | None = None,
) -> bool:
    """
    Install Torch suite packages sequentially using a single locked index.
    If the suggested index fails, prompt the user for alternatives.
    """
    # Start with the best suggested index
    current_idx = suggested_indices[0] if suggested_indices else None

    # Step 1: Attempt the locked install
    while True:
        console.print(f"   Using locked index: [cyan]{current_idx or 'Standard PyPI'}[/cyan]")
        success = True

        for req in reqs:
            solved_ver = (solved_map or {}).get(req.name)
            req_str = f"{req.name}=={solved_ver}" if solved_ver and not req.is_url else str(req)

            # Build environment if it's a heavy package (e.g. xformers)
            pkg_env = os.environ.copy()
            if req.name in HEAVY_COMPILERS:
                config = HEAVY_COMPILERS[req.name]
                pkg_env = get_build_env(toolchain, config, req.name)

            cmd = ["uv", "pip", "install", "--python", python_executable, req_str]
            if current_idx:
                cmd.extend(["--index-url", current_idx])

            console.print(f"   -> [dim]Installing {req_str}...[/dim]")
            try:
                subprocess.run(cmd, env=pkg_env, check=True)
            except subprocess.CalledProcessError:
                print_error(f"Failed to install {req.name} via {current_idx or 'PyPI'}")
                success = False
                break
        
        if success:
            return True

        # Handle failure: Interactive prompt
        console.print()
        print_warning("The current CUDA index failed to resolve the Batch.")
        console.print("   Possible alternatives:")
        for i, idx in enumerate(suggested_indices):
            marker = "<- Current" if idx == current_idx else ""
            console.print(f"     [{i}] {idx} {marker}")
        console.print(f"     [s] Skip Torch Suite (Not recommended)")
        
        print("\n   Select index number or 's' to skip: ", end="", flush=True)
        try:
            choice = input("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "s"

        if choice == "s":
            return False
        
        try:
            idx_choice = int(choice)
            if 0 <= idx_choice < len(suggested_indices):
                current_idx = suggested_indices[idx_choice]
                continue
        except ValueError:
            pass
        
        print_error("Invalid choice. Please try again.")
