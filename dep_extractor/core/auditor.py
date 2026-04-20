"""
dep_extractor/core/auditor.py
==============================
Phase 2 — GINGER-standard hardware and environment audit.

Educational: This module answers ONE question: "What hardware and software
environment is this script running in?"

The result (AuditData) is used throughout downstream phases to:
  - Select the correct CUDA PyTorch index URL
  - Choose the right Python target version for uv pip compile
  - Gate heavy-compiler workflows on MSVC availability
  - Report VRAM capacity for model compatibility warnings

Key improvements over the monolith:
  - All probing functions are independently testable (easy to mock subprocess)
  - suggest_best_indices() uses sorted(CUDA_INDEX_MAP) instead of cascading ifs
  - Returns AuditData dataclass instead of an opaque dict
  - nvidia-smi and winreg calls are isolated into private helpers
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

from dep_extractor.config import CUDA_INDEX_MAP
from dep_extractor.models import AuditData, GpuInfo, InstalledPackage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_hardware_audit(python_executable: str | None = None) -> AuditData:
    """
    Execute the full GINGER-standard hardware and environment probe.

    This is the primary entry point for Phase 2. It runs all sub-probes
    and assembles a complete AuditData struct.

    Returns:
        AuditData — fully populated on success, using safe defaults
        for any sub-probe that fails (EAFP pattern: log + continue).
    """
    audit = AuditData()
    if python_executable:
        audit.python_executable = str(Path(python_executable).resolve())

    # UV detection (The 'SAT Engine' prerequisite)
    audit.has_uv, audit.uv_path = _probe_uv()

    # Terminal detection
    audit.terminal = os.environ.get("TERM_PROGRAM", "Unknown")

    # OS-specific probes
    if audit.os == "Windows":
        _apply_windows_probes(audit)

    logger.info(
        "Audit complete — OS: %s, CUDA: %s, GPUs: %d, uv: %s",
        audit.os,
        audit.cuda_version,
        len(audit.gpus),
        audit.has_uv,
    )
    return audit


def get_installed_packages(python_executable: str | None = None) -> list[InstalledPackage]:
    """
    Retrieve the current pip-installed package inventory.

    Runs `pip list --format=json` in the specified or active interpreter.

    Args:
        python_executable: Optional path to a specific python to probe.
                          Defaults to sys.executable if None.

    Returns:
        list[InstalledPackage]: Typed list of installed packages.
        Returns empty list on failure (logs warning — does not crash).
    """
    python_exe = python_executable or sys.executable
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw: list[dict] = json.loads(result.stdout)
        packages = [InstalledPackage(name=p["name"], version=p["version"]) for p in raw]
        logger.info("Found %d packages installed in current venv.", len(packages))
        return packages
    except Exception as exc:
        logger.warning("Failed to get installed packages: %s", exc)
        return []


def detect_index_urls(packages: list[InstalledPackage]) -> list[str]:
    """
    Scan installed packages for hardware-specific suffixes to infer needed
    PyTorch wheel index URLs.

    Educational: torch packages carry version suffixes like `+cu124` or
    `+rocm5.4` encoding which CUDA/ROCm they were built against. By detecting
    these we can reproduce the same index URLs that originally installed them.

    Args:
        packages: The output of get_installed_packages().

    Returns:
        Sorted list of detected PyTorch hardware index URLs.
    """
    indices: set[str] = set()
    for pkg in packages:
        version = pkg.version
        if "+cu" in version:
            cu = re.search(r"cu(\d+)", version)
            if cu:
                idx = f"https://download.pytorch.org/whl/{cu.group(0)}"
                if idx not in indices:
                    logger.info("Detected CUDA index from %s: %s", pkg.name, idx)
                indices.add(idx)
        if "+rocm" in version:
            rocm = re.search(r"rocm(\d+\.\d+)", version)
            if rocm:
                idx = f"https://download.pytorch.org/whl/{rocm.group(0)}"
                if idx not in indices:
                    logger.info("Detected ROCm index from %s: %s", pkg.name, idx)
                indices.add(idx)
    return sorted(indices)


def suggest_best_indices(audit: AuditData) -> list[str]:
    """
    Map detected CUDA version to the highest compatible PyTorch index URLs.

    Educational: Instead of the monolith's cascading `if v >= 12.8... if v >= 12.6...`
    chain, we use `sorted(CUDA_INDEX_MAP, reverse=True)` to iterate cleanly.
    All CUDA versions up to and including the detected version are included,
    so that the highest-compatible index is tried first (uv/pip uses first-match).

    Args:
        audit: Populated AuditData from run_hardware_audit().

    Returns:
        List of recommended PyTorch index URLs (highest-compat first).
    """
    indices: list[str] = []

    if audit.cuda_version:
        try:
            v_float = float(audit.cuda_version)
            for threshold in sorted(CUDA_INDEX_MAP.keys(), reverse=True):
                if v_float >= threshold:
                    indices.append(CUDA_INDEX_MAP[threshold])
        except (ValueError, TypeError):
            logger.warning("Could not parse CUDA version: %s", audit.cuda_version)
    elif any(g.gpu_type == "NVIDIA" for g in audit.gpus):
        # NVIDIA GPU detected but nvidia-smi couldn't report version — safe default
        logger.info("NVIDIA GPU detected but CUDA version unknown — using cu121 fallback.")
        indices.append(CUDA_INDEX_MAP.get(12.1, "https://download.pytorch.org/whl/cu121"))

    # Deduplicate while preserving order
    return list(dict.fromkeys(indices))


# ---------------------------------------------------------------------------
# Private sub-probes
# ---------------------------------------------------------------------------

def _probe_uv() -> tuple[bool, str | None]:
    """
    Checks whether `uv` is available in PATH.

    Returns:
        (has_uv: bool, uv_path: str | None)
    """
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "System PATH"
    except FileNotFoundError:
        pass
    return False, None


def _apply_windows_probes(audit: AuditData) -> None:
    """
    Runs all Windows-specific hardware probes in-place on `audit`.
    Mutates audit directly (speed: avoids creating intermediate dicts).
    """
    # 1. Windows Insider detection via registry
    _probe_windows_registry(audit)

    # 2. NVIDIA GPU / CUDA via nvidia-smi (primary source)
    gpus, cuda_ver = _probe_nvidia_smi()
    if gpus:
        audit.gpus = gpus
        audit.cuda_version = cuda_ver
        audit.total_vram_gb = sum(g.vram_gb for g in gpus)
        audit.is_ginger_class = any("4090" in g.name or "5090" in g.name for g in gpus)
        if cuda_ver:
            logger.info("Detected CUDA Version: %s", cuda_ver)

    # 3. CIM fallback when nvidia-smi is missing (non-NVIDIA or driver issue)
    if not audit.gpus:
        fallback_gpus = _probe_cim_gpu()
        audit.gpus = fallback_gpus
        audit.total_vram_gb = sum(g.vram_gb for g in fallback_gpus)


def _probe_windows_registry(audit: AuditData) -> None:
    """
    Reads Windows build number from the registry to detect Insider builds.

    Educational: Windows Insider Preview builds report a CurrentBuild value
    higher than 22621 (Windows 11 22H2). We use this to select Python 3.13
    as the uv compilation target (Insider supports newer runtimes).
    """
    try:
        import winreg  # type: ignore[import]
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        )
        edition, _ = winreg.QueryValueEx(key, "EditionID")
        build_str, _ = winreg.QueryValueEx(key, "CurrentBuild")
        if int(build_str) > 22621:
            audit.is_insider = True
        logger.debug("Windows edition: %s, build: %s, insider: %s", edition, build_str, audit.is_insider)
    except Exception as exc:
        logger.debug("winreg probe skipped: %s", exc)


def _probe_nvidia_smi() -> tuple[list[GpuInfo], str | None]:
    """
    Queries nvidia-smi for GPU names, VRAM, and CUDA version.

    Returns:
        (gpus: list[GpuInfo], cuda_version: str | None)
        Empty list if nvidia-smi is unavailable or fails.
    """
    gpus: list[GpuInfo] = []
    cuda_version: str | None = None

    try:
        # Query 1: GPU names and VRAM
        smi_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if smi_query.returncode == 0:
            for line in smi_query.stdout.strip().splitlines():
                parts = line.split(",", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        vram_gb = int(parts[1].strip()) // 1024
                    except ValueError:
                        vram_gb = 0
                    gpus.append(GpuInfo(gpu_type="NVIDIA", name=name, vram_gb=vram_gb))

        # Query 2: CUDA version from the nvidia-smi banner
        smi_banner = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
        )
        if smi_banner.returncode == 0:
            cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_banner.stdout)
            if cuda_match:
                cuda_version = cuda_match.group(1)

    except FileNotFoundError:
        logger.debug("nvidia-smi not found — NVIDIA not detected.")
    except Exception as exc:
        logger.warning("nvidia-smi probe failed: %s", exc)

    return gpus, cuda_version


def _probe_cim_gpu() -> list[GpuInfo]:
    """
    Fallback GPU probe via PowerShell CIM (Win32_VideoController).

    Used when nvidia-smi is not available (e.g., non-NVIDIA GPU, driver uninstalled).
    This surfaces AMD, Intel Arc, and integrated graphics.

    Returns:
        list[GpuInfo] — empty if probe fails.
    """
    try:
        cmd = (
            "Get-CimInstance Win32_VideoController "
            "| Select-Object Name,AdapterRAM "
            "| ConvertTo-Json"
        )
        ps_out = subprocess.run(
            ["pwsh", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
        )
        if ps_out.returncode != 0 or not ps_out.stdout.strip():
            return []

        data = json.loads(ps_out.stdout)
        if isinstance(data, dict):
            data = [data]

        gpus = []
        for gpu in data:
            name = gpu.get("Name", "Unknown")
            vram_raw = gpu.get("AdapterRAM") or 0
            vram_gb = vram_raw // (1024 ** 3)
            gpus.append(GpuInfo(gpu_type="Other", name=name, vram_gb=vram_gb))
        return gpus

    except Exception as exc:
        logger.debug("CIM GPU fallback failed: %s", exc)
        return []
