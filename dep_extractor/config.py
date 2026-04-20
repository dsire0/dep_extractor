"""
dep_extractor/config.py
=======================
All global constants and registries for the ComfyUI Dependency Resolver Suite.

Educational: Centralising constants into a single config module prevents
"magic strings" — hard-coded values buried inside logic functions — which
are a primary cause of silent misconfiguration bugs.

This module replaces the scattered constants at the top of the original
requirements_extractor.py monolith (HEAVY_COMPILERS dict, PLATFORM_REMAPS,
CUDA index cascades) with a structured, auditable registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Heavy Compiler Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeavyCompilerConfig:
    """
    Descriptor for a package that requires native compilation toolchain.

    These packages cannot be installed from a pre-built wheel on Windows
    without MSVC, Ninja, and sometimes CMake. Each entry describes
    exactly what is needed and what environment variables to inject.

    Field notes:
      - `requires_msvc`   : Needs cl.exe (Visual C++ compiler)
      - `requires_ninja`  : Needs ninja.exe (fast build orchestrator)
      - `requires_cmake`  : Needs cmake.exe (build system generator)
      - `env_vars`        : Environment overrides to inject at build time
      - `wheel_sources`   : Pre-built wheel URL templates (version/cuda/python)
    """
    requires_msvc: bool = False
    requires_ninja: bool = False
    requires_cmake: bool = False
    env_vars: dict = field(default_factory=dict)
    wheel_sources: list = field(default_factory=list)


# The canonical heavy compiler registry.
# Key = normalized package name (lowercase, hyphenated).
HEAVY_COMPILERS: dict[str, HeavyCompilerConfig] = {
    "deepspeed": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=False,
        env_vars={
            "DS_BUILD_AIO": "0",          # AIO never supported on Windows
            "DISTUTILS_USE_SDK": "1",
        },
    ),
    "flash-attn": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=True,
        env_vars={
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "MAX_JOBS": "4",
            "DISTUTILS_USE_SDK": "1",
        },
        wheel_sources=[
            # Template vars: {version}, {cuda}, {torch}, {python}
            "https://github.com/bdashore3/flash-attention/releases/download/"
            "v{version}/flash_attn-{version}+cu{cuda}torch{torch}"
            "cxx11abiFALSE-cp{python}-cp{python}-win_amd64.whl"
        ],
    ),
    "sageattention": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=False,
        env_vars={"DISTUTILS_USE_SDK": "1"},
    ),
    "xformers": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=False,
        env_vars={
            "XFORMERS_MORE_DETAILS": "1",
            "DISTUTILS_USE_SDK": "1",
        },
    ),
    "llama-cpp-python": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=True,
        env_vars={
            "CMAKE_ARGS": "-DGGML_CUDA=on", # Default to CUDA build
            "FORCE_CMAKE": "1",
            "DISTUTILS_USE_SDK": "1",
        },
    ),
    "insightface": HeavyCompilerConfig(
        requires_msvc=True,
        requires_ninja=True,
        requires_cmake=False,
    ),
    "dlib": HeavyCompilerConfig(
        requires_msvc=True,
        requires_cmake=True,
    ),
    "sentencepiece": HeavyCompilerConfig(
        requires_msvc=True,
        requires_cmake=True,
    ),
}


# ---------------------------------------------------------------------------
# Platform Package Remaps
# ---------------------------------------------------------------------------

# Educational: Some packages have OS-specific replacements.
# 'triton' only has a Windows port under the 'triton-windows' name.
# This registry is applied during normalization to silently swap
# the incompatible package name for the correct platform equivalent.
PLATFORM_REMAPS: dict[str, dict[str, str]] = {
    "Windows": {
        "triton": "triton-windows",
    },
    "Linux": {},
    "Darwin": {},
}


# ---------------------------------------------------------------------------
# GPU Preference Overrides
# ---------------------------------------------------------------------------

# Educational: Some packages have GPU and CPU variants (e.g. onnxruntime).
# When both are present in the combined requirements, the GPU version
# is always preferred to avoid silently falling back to CPU-only inference.
GPU_PREFS: dict[str, str] = {
    "onnxruntime": "onnxruntime-gpu",
}


# ---------------------------------------------------------------------------
# CUDA PyTorch Index URL Map
# ---------------------------------------------------------------------------

# Maps minimum CUDA version (float) → highest compatible PyTorch wheel index.
# Educational: PyTorch distributes hardware-specific binaries through
# "extra index URLs". Without the correct URL, pip/uv defaults to a
# generic CPU-only build, which silently disables CUDA acceleration.
#
# Usage: iterate in descending key order; first match wins.
CUDA_INDEX_MAP: dict[float, str] = {
    13.0: "https://download.pytorch.org/whl/cu130",
    12.8: "https://download.pytorch.org/whl/cu128",
    12.6: "https://download.pytorch.org/whl/cu126",
    12.4: "https://download.pytorch.org/whl/cu124",
    12.1: "https://download.pytorch.org/whl/cu121",
    11.8: "https://download.pytorch.org/whl/cu118",
}


# ---------------------------------------------------------------------------
# Environment Stabilisation Constants
# ---------------------------------------------------------------------------

# These mirror the env-var block at the top of the original monolith.
# They are applied once by cli.py at startup.
ENV_STABILISATION: dict[str, str] = {
    "DISTUTILS_USE_SDK": "1",
    "FLET_SERVER_PORT": "8550",
    "FLET_SERVER_IP": "127.0.0.1",
}


# ---------------------------------------------------------------------------
# Packages excluded from "missing from requirements" audit
# ---------------------------------------------------------------------------

# These are always present in a Python venv and are not tracked by any
# custom node's requirements.txt — that is expected and normal.
BASELINE_PACKAGES: frozenset[str] = frozenset({
    "pip", "setuptools", "wheel", "uv", "packaging",
    "distlib", "filelock", "platformdirs", "virtualenv",
})
