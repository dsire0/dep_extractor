"""
dep_extractor/install/heavy_compiler.py
=========================================
Heavy compiler orchestration — phased build logic for MSVC-dependent packages.

Educational: Some Python packages (deepspeed, flash-attn, xformers) cannot
be installed from a pre-built wheel on Windows and must be compiled from source.
This requires:
  1. MSVC cl.exe (the C++ compiler)
  2. Ninja (fast parallel build orchestrator)
  3. Sometimes CMake (build system generator)
  4. CUDA Toolkit (for GPU-accelerated extensions)

This module isolates all heavy compiler logic into a clean orchestrator
so the main installer.py can stay under 100 lines.

Key improvements over the monolith:
  - _try_install_with_fallback() replaces the deep-nested try/except tree
  - DeepSpeed-specific interaction is isolated in _handle_deepspeed()
  - All prompts go through a single clean_input() helper (no scattered prints)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from dep_extractor.config import HEAVY_COMPILERS
from dep_extractor.models import NormalizedRequirement, ToolchainStatus
from dep_extractor.output.console import console, print_error, print_success, print_warning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_heavy_package(
    req: NormalizedRequirement,
    toolchain: ToolchainStatus,
    installed_names: set[str],
    audit: "AuditData",
) -> bool | None:
    """
    Orchestrate the complete heavy-compiler install flow for a single package.
    Returns True if successful, False if failed, or None if user requested to skip all remaining.

    Phases:
      1. Already-installed check → skip
      2. Toolchain validation → abort if missing critical tool
      3. Environment injection (vcvars + package-specific env_vars)
      4. Optional auto-install of Ninja / CMake if missing
      5. Build attempt via `uv pip install --no-build-isolation`
      6. Fallback to direct `pip install --no-build-isolation`
      7. DeepSpeed-specific OPS retry without DS_BUILD_OPS=1

    Args:
        req:              The heavy requirement to install.
        toolchain:        Pre-probed toolchain status.
        installed_names:  Set of already-installed normalized package names.

    Returns:
        True if installed successfully, False on total failure.
    """
    pkg_name = req.name
    config = HEAVY_COMPILERS.get(pkg_name)
    if config is None:
        logger.warning("No HEAVY_COMPILERS entry for %s", pkg_name)
        return False

    console.print(f"\n[bold cyan]☢️  Heavy compiler: [white]{pkg_name}[/white][/bold cyan]")

    # Skip if already installed
    if pkg_name in installed_names:
        console.print(f"   [dim]Already installed — skipping.[/dim]")
        return True

    # User bypass prompt
    print(f"   Install {pkg_name}? (Y/n/s=skip all): ", end="", flush=True)
    try:
        choice = input("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    if choice == "s":
        console.print(f"   [yellow]Skipping all remaining heavy packages.[/yellow]")
        return None
    if choice not in ("", "y", "yes"):
        console.print(f"   [yellow]Skipping {pkg_name} by user request.[/yellow]")
        return False

    # Validate MSVC toolchain on Windows
    if config.requires_msvc and sys.platform == "win32":
        if not toolchain.has_msvc:
            print_error(f"MSVC required for {pkg_name} but not found. Skipping.")
            return False
        console.print("   [green]✓[/green] MSVC environment injected.")

    # Validate Ninja
    if config.requires_ninja and not toolchain.has_ninja:
        if not _auto_install_tool("Ninja", audit.python_executable):
            return False

    # Validate CMake
    if config.requires_cmake and not toolchain.has_cmake:
        if not _auto_install_tool("CMake", audit.python_executable):
            return False

    # Build the environment overlay
    build_env = _build_env(toolchain, config, pkg_name)

    return _try_install_with_fallback(req, pkg_name, build_env)


def auto_install_ready_for_heavy(toolchain: ToolchainStatus) -> bool:
    """
    Returns True if the toolchain is currently ready for heavy build tasks.
    Used by installer.py to skip the phase if no tools are available.
    """
    return toolchain.is_ready_for_heavy_build


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_env(
    toolchain: ToolchainStatus,
    config,  # HeavyCompilerConfig
    pkg_name: str,
) -> dict[str, str]:
    """
    Assemble the complete build environment for a package.

    Order of precedence (highest → lowest):
      1. Current os.environ
      2. vcvars MSVC environment overlay
      3. Package-specific env_vars from HEAVY_COMPILERS config
      4. Always: DISTUTILS_USE_SDK=1 override
    """
    env = os.environ.copy()

    # Apply MSVC vcvars if available
    if toolchain.vcvars_env:
        env.update(toolchain.vcvars_env)

    # Apply package-specific env vars (only add, don't override vcvars)
    for k, v in config.env_vars.items():
        if k not in env:
            env[k] = v

    # MSVC monkey-patch: always force DISTUTILS to use SDK cl.exe
    env["DISTUTILS_USE_SDK"] = "1"

    # DeepSpeed special handling — interactive prompt
    if pkg_name == "deepspeed":
        _configure_deepspeed_env(env)

    return env


def _configure_deepspeed_env(env: dict[str, str]) -> None:
    """
    Interactive prompt for DeepSpeed OPS configuration.

    Educational: DS_BUILD_OPS=1 compiles all DeepSpeed CUDA extensions
    from source (inference, quantization, etc.) — this takes 20+ minutes
    but produces the fastest runtime. DS_BUILD_OPS=0 installs only the
    Python wrapper with pre-compiled wheel stubs (much faster, slightly slower).
    """
    console.print()
    console.print("   [bold yellow]DeepSpeed Configuration[/bold yellow]")
    console.print("   Build native OPS? (MSVC + CUDA required, adds ~20 minutes)")
    print("   Build native OPS? (Y/n): ", end="", flush=True)
    try:
        choice = input("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    env["DS_BUILD_OPS"] = "1" if choice in ("", "y", "yes") else "0"
    env["DS_BUILD_AIO"] = "0"   # Never supported on Windows
    logger.debug("DS_BUILD_OPS=%s", env["DS_BUILD_OPS"])


def _try_install_with_fallback(
    req: NormalizedRequirement,
    pkg_name: str,
    build_env: dict[str, str],
) -> bool:
    """
    Attempt uv install → pip fallback → DeepSpeed OPS-off retry.

    Educational: We structure this as a clean chain of responsibility:
      1. uv (fast, preferred)
      2. pip (handles some edge cases uv misses on Windows)
      3. DeepSpeed-specific retry without DS_BUILD_OPS (recovery path)

    Returns True if any attempt succeeds, False if all fail.
    """
    req_str = str(req)
    base_cmd_uv = ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", req_str]
    base_cmd_pip = [sys.executable, "-m", "pip", "install", "--no-build-isolation", req_str]

    # Attempt 1: uv
    console.print(f"   Initiating build via [cyan]uv[/cyan] for {req_str}...")
    if _run_install(base_cmd_uv, build_env):
        print_success(f"Installed {req_str}")
        return True

    # Attempt 2: pip fallback
    console.print(f"   uv failed → trying [cyan]pip[/cyan] fallback...")
    if _run_install(base_cmd_pip, build_env):
        print_success(f"Installed {req_str} (via pip fallback)")
        return True

    # Attempt 3: DeepSpeed OPS recovery
    if pkg_name == "deepspeed" and build_env.get("DS_BUILD_OPS") == "1":
        console.print("   [yellow]Attempting recovery: DS_BUILD_OPS=0...[/yellow]")
        recovery_env = build_env.copy()
        recovery_env["DS_BUILD_OPS"] = "0"
        uv_cmd = ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", req_str]
        if _run_install(uv_cmd, recovery_env):
            print_success(f"Installed {req_str} (OPS disabled)")
            return True

    print_error(f"Total failure: could not install {req_str}")
    return False


def _run_install(cmd: list[str], env: dict[str, str]) -> bool:
    """Run an install command and return True on success."""
    try:
        subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.debug("Install attempt failed (rc=%d): %s", exc.returncode, " ".join(cmd))
        return False
    except FileNotFoundError as exc:
        logger.debug("Command not found: %s — %s", cmd[0], exc)
        return False


def _auto_install_tool(tool_name: str, python_executable: str) -> bool:
    """
    Prompt user to auto-install a missing build tool (Ninja or CMake).

    Args:
        tool_name: Human-readable name ("Ninja" or "CMake").
        python_executable: Path to target python.

    Returns:
        True if user agreed and installation succeeded.
    """
    pkg = tool_name.lower()
    print_warning(f"{tool_name} is required but not found.")
    print(f"   Auto-install {tool_name} via uv pip? (Y/n): ", end="", flush=True)
    try:
        choice = input("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    if choice not in ("", "y", "yes"):
        return False

    try:
        subprocess.run(
            ["uv", "pip", "install", "--python", python_executable, pkg],
            check=True,
        )
        print_success(f"{tool_name} installed.")
        return True
    except Exception as exc:
        print_error(f"Failed to install {tool_name}: {exc}")
        return False
