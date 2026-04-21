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

from dep_extractor.config import HEAVY_COMPILERS, LLAMA_CU_INDEX_MAP
from dep_extractor.models import NormalizedRequirement, ToolchainStatus, _version_satisfies
from dep_extractor.output.console import console, print_error, print_success, print_warning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_heavy_package(
    req: NormalizedRequirement,
    toolchain: ToolchainStatus,
    installed_map: dict[str, str],
    audit: "AuditData",
) -> bool | None:
    """
    Orchestrate the complete heavy-compiler install flow for a single package.
    Returns True if successful, False if failed, or None if user requested to skip all remaining.

    Phases:
      0. Already-satisfied check → skip immediately (silent)
      1. Toolchain validation → abort if missing critical tool
      2. Environment injection (vcvars + package-specific env_vars)
      3. Dynamic variant labeling ([GPU/CUDA], [GPU/ROCm], etc.)
      4. Optional auto-install of Ninja / CMake if missing
      5. Build attempt via `uv pip install --no-build-isolation --no-binary :all:`
      6. Fallback to direct `pip install --no-build-isolation`
      7. DeepSpeed-specific OPS retry without DS_BUILD_OPS=1

    Args:
        req:              The heavy requirement to install.
        toolchain:        Pre-probed toolchain status.
        installed_map:    Map of normalized_name -> installed version.
        audit:            Populated AuditData.

    Returns:
        True if installed successfully, False on total failure.
    """
    pkg_name = req.name
    
    # ----------------------------------------------------------------------
    # Phase 0: Silent satisfaction check
    # ----------------------------------------------------------------------
    # Educational: We check this before ANY console output to avoid the 
    # "False Positive" warnings for packages already handled or satisfied.
    inst_ver = installed_map.get(pkg_name)
    if inst_ver and _version_satisfies(inst_ver, req.specifier):
        logger.debug("Heavy package %s already satisfied (version=%s)", pkg_name, inst_ver)
        return True

    config = HEAVY_COMPILERS.get(pkg_name)
    if config is None:
        logger.warning("No HEAVY_COMPILERS entry for %s", pkg_name)
        return False

    # Variant determination for UI
    variant = _get_variant_label(config)
    variant_text = f" [bold magenta]{variant}[/bold magenta]" if variant else ""

    console.print(f"\n[bold cyan]☢️  Heavy compiler: [white]{pkg_name}[/white]{variant_text}[/bold cyan]")
    if pkg_name in ("llama-cpp-python", "flash-attn", "deepspeed"):
        console.print("   [bold yellow]⚠️  WARNING: This package is known to take 10-30 minutes to build.[/bold yellow]")
        if pkg_name == "llama-cpp-python":
            console.print("   [dim]Tip: If you want a pre-built wheel, check the official abetlen/llama-cpp-python releases.[/dim]")

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
    build_env = get_build_env(toolchain, config, pkg_name)

    return _try_install_with_fallback(req, pkg_name, build_env, audit)


def auto_install_ready_for_heavy(toolchain: ToolchainStatus) -> bool:
    """
    Returns True if the toolchain is currently ready for heavy build tasks.
    Used by installer.py to skip the phase if no tools are available.
    """
    return toolchain.is_ready_for_heavy_build


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def get_build_env(
    toolchain: ToolchainStatus,
    config: "HeavyCompilerConfig",
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

    # Apply package-specific env vars (CRITICAL: Hard override to ensure build flags apply)
    # Educational: We use a hard override here because the user may have generic
    # environment variables set in their shell that would otherwise block
    # build-critical flags (like -DGGML_CUDA=on) if we checked for 'k not in env'.
    for k, v in config.env_vars.items():
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
    audit: "AuditData",
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
    # Educational: We use --no-binary :all: for heavy compilers to ENSURE that
    # any injected environment variables (like -DGGML_CUDA=on) are actually 
    # used by forcing a source build, even if a CPU wheel exists on PyPI.
    base_cmd_uv = [
        "uv", "pip", "install", 
        "--python", audit.python_executable, 
        "--no-build-isolation", 
        "--no-binary", pkg_name, # Only for THIS package to avoid breaking other deps
        req_str
    ]
    base_cmd_pip = [
        audit.python_executable, "-m", "pip", "install", 
        "--no-build-isolation", 
        "--no-binary", pkg_name,
        req_str
    ]

    # Pre-built wheels optimization for llama-cpp-python
    if pkg_name == "llama-cpp-python" and audit.cuda_version:
        try:
            v_float = float(audit.cuda_version)
            for threshold in sorted(LLAMA_CU_INDEX_MAP.keys(), reverse=True):
                if v_float >= threshold:
                    index_url = LLAMA_CU_INDEX_MAP[threshold]
                    console.print(f"   [bright_green]Found valid pre-built wheel index:[/bright_green] {index_url}")
                    # Allow binary from the extra index url (removes --no-binary pkg_name)
                    base_cmd_uv.remove("--no-binary")
                    base_cmd_uv.remove(pkg_name)
                    base_cmd_pip.remove("--no-binary")
                    base_cmd_pip.remove(pkg_name)
                    
                    base_cmd_uv.extend(["--extra-index-url", index_url, "--index-strategy", "unsafe-best-match"])
                    base_cmd_pip.extend(["--extra-index-url", index_url])
                    break
        except (ValueError, TypeError):
            pass

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
        uv_cmd = ["uv", "pip", "install", "--python", audit.python_executable, "--no-build-isolation", req_str]
        if _run_install(uv_cmd, recovery_env):
            print_success(f"Installed {req_str} (OPS disabled)")
            return True

    print_error(f"Total failure: could not install {req_str}")
    return False


def _get_variant_label(config: "HeavyCompilerConfig") -> str | None:
    """
    Inspect env_vars to determine the build variant (GPU, ROCm, etc.).
    """
    env_str = str(config.env_vars).upper()
    cmake_args = config.env_vars.get("CMAKE_ARGS", "").upper()
    
    # Generic GPU/CUDA detection
    if any(k in env_str or k in cmake_args for k in ("CUDA", "CUBLAS", "NVCC", "NVIDIA")):
        return "[GPU/CUDA]"
    
    # AMD/ROCm detection
    if any(k in env_str or k in cmake_args for k in ("ROCM", "HIP", "AMD")):
        return "[GPU/ROCm]"
    
    # Metal/Apple detection
    if "METAL" in env_str or "METAL" in cmake_args:
        return "[GPU/METAL]"
        
    return None


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
