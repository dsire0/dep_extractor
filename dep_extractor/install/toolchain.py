"""
dep_extractor/install/toolchain.py
====================================
Build toolchain detection and validation for heavy compiler packages.

Educational: Packages like deepspeed and flash-attn require a complete
C++ build toolchain to compile from source on Windows. This module
isolates ALL toolchain detection logic so it can be:
  1. Tested independently (mock subprocess calls)
  2. Called eagerly once (not per-package as in the monolith)
  3. Returned as a typed ToolchainStatus — not scattered boolean flags

The MSVC detection uses vswhere.exe, which is Microsoft's officially
supported tool for locating Visual Studio installations — it's always
available at a fixed path if Visual Studio is installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from dep_extractor.models import ToolchainStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_toolchain() -> ToolchainStatus:
    """
    Probe the local build toolchain completeness in a single pass.

    Runs all detection functions and assembles a ToolchainStatus struct.
    This is called ONCE at the start of the install phase — the result is
    passed to each heavy compiler handler so per-package detection is avoided.

    Returns:
        ToolchainStatus with all available build capabilities populated.
    """
    status = ToolchainStatus()

    # MSVC detection (via vswhere.exe — only relevant on Windows)
    has_msvc, vs_path, msvc_error = detect_msvc()
    status.has_msvc = has_msvc
    status.vs_install_path = vs_path
    if msvc_error and has_msvc is False:
        logger.debug("MSVC not available: %s", msvc_error)

    # Load vcvars environment if MSVC found
    if has_msvc and vs_path:
        env, err = load_vcvars_env(vs_path)
        if env:
            status.vcvars_env = env
            logger.info("MSVC vcvars environment loaded (%d vars).", len(env))
        elif err:
            logger.warning("vcvars load failed: %s", err)

    # Simple PATH checks for Ninja and CMake
    status.has_ninja = shutil.which("ninja") is not None
    status.has_cmake = shutil.which("cmake") is not None

    # CUDA Toolkit detection
    status.has_cuda_toolkit, _ = detect_cuda_toolkit()

    logger.info(
        "Toolchain probe — MSVC: %s, Ninja: %s, CMake: %s, CUDA toolkit: %s",
        status.has_msvc,
        status.has_ninja,
        status.has_cmake,
        status.has_cuda_toolkit,
    )
    return status


def detect_msvc() -> tuple[bool, str | None, str | None]:
    """
    Detect Microsoft Visual C++ Build Tools using vswhere.exe.

    Educational: vswhere.exe is always installed to a fixed path when VS is
    present. It's the only reliable way to find VS across non-standard install
    locations (e.g., D: drive installs, Preview channels, etc.).

    Returns:
        (has_msvc: bool, vs_install_path: str | None, error: str | None)
    """
    program_files_x86 = os.environ.get(
        "ProgramFiles(x86)", r"C:\Program Files (x86)"
    )
    vswhere = os.path.join(
        program_files_x86,
        "Microsoft Visual Studio",
        "Installer",
        "vswhere.exe",
    )

    if not os.path.exists(vswhere):
        return False, None, "vswhere.exe not found — Visual Studio may not be installed."

    try:
        result = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        path = result.stdout.strip()
        if path:
            return True, path, None
        return False, None, "MSVC C++ Build Tools workload not found in any VS install."
    except subprocess.CalledProcessError as exc:
        return False, None, f"vswhere failed: {exc.stderr}"
    except Exception as exc:
        return False, None, str(exc)


def detect_cuda_toolkit() -> tuple[bool, str | None]:
    """
    Detect the CUDA Toolkit (nvcc compiler, not just the driver).

    Educational: There are two different CUDA things:
      1. The NVIDIA driver (exposes CUDA runtime, detected by nvidia-smi)
      2. The CUDA Toolkit (includes nvcc, required for compilation)

    This function checks for the Toolkit specifically.

    Returns:
        (has_cuda_toolkit: bool, path: str | None)
    """
    # Check CUDA_PATH environment variable first (most reliable on Windows)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.isdir(cuda_path):
        return True, cuda_path

    # Fallback: check for nvcc in PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        return True, os.path.dirname(os.path.dirname(nvcc))

    return False, None


def load_vcvars_env(vs_install_path: str) -> tuple[dict[str, str] | None, str | None]:
    """
    Capture the complete MSVC C++ build environment from vcvarsall.bat.

    Educational: MSVC requires dozens of environment variables (INCLUDE, LIB,
    PATH, etc.) to find cl.exe, the SDK headers, and linker libraries. Rather
    than guessing these paths, we call vcvarsall.bat and capture its output
    environment via a `set` dump — the same technique that MSVC build systems use.

    Args:
        vs_install_path: The Visual Studio installation root path.

    Returns:
        (env_dict: dict | None, error: str | None)
        On success, env_dict contains all MSVC build environment variables.
    """
    # Try vcvars64.bat first (direct x64), then vcvarsall.bat x64
    vcvars_candidates = [
        os.path.join(vs_install_path, "VC", "Auxiliary", "Build", "vcvars64.bat"),
        os.path.join(vs_install_path, "VC", "Auxiliary", "Build", "vcvarsall.bat"),
    ]

    vcvars_path = next((p for p in vcvars_candidates if os.path.exists(p)), None)
    if not vcvars_path:
        return None, "vcvars64.bat / vcvarsall.bat not found in VS installation."

    try:
        # Call vcvars and immediately dump the environment with `set`
        arg = "x64" if "vcvarsall" in vcvars_path else ""
        cmd = f'"{vcvars_path}" {arg} && set'.strip()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
        )

        if result.returncode != 0:
            return None, f"vcvars execution failed: {result.stderr[:200]}"

        # Parse the `set` output into a dict
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env[key] = value

        return env, None

    except Exception as exc:
        return None, str(exc)
