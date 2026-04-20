"""
dep_extractor/utils/wheel_parser.py
===================================
Utility for parsing PEP 427 wheel filenames and checking platform compatibility.
"""

from __future__ import annotations
import re
from pathlib import Path
from dep_extractor.models import WheelMetadata, AuditData

# PEP 427 tag priority: name-version(-build)?-python-abi-platform.whl
WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?-"
    r"(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^.]+)\.whl$",
    re.IGNORECASE
)

def parse_wheel_filename(filename: str) -> WheelMetadata | None:
    """
    Parses a .whl filename into its component tags.
    Returns WheelMetadata or None if the filename is not a valid wheel name.
    """
    # Normalize by taking only the filename if a path or URL was passed
    name_only = filename.split("/")[-1].split("\\")[-1]
    
    match = WHEEL_RE.match(name_only)
    if not match:
        return None
        
    gd = match.groupdict()
    return WheelMetadata(
        name=gd["name"],
        version=gd["version"],
        build=gd["build"],
        python_tags=gd["python"].split("."),
        abi_tags=gd["abi"].split("."),
        platform_tags=gd["platform"].split("."),
    )

def check_wheel_compatibility(metadata: WheelMetadata, audit: AuditData) -> tuple[bool, str | None]:
    """
    Checks if a wheel is compatible with the audited environment.
    Returns (is_compatible, reason).
    """
    # 1. Platform Check
    os_map = {
        "Windows": "win",
        "Linux": "linux",
        "Darwin": "macosx",
    }
    target_os_prefix = os_map.get(audit.os, "unknown")
    
    # Check if any platform tag matches the target OS
    platform_match = False
    for tag in metadata.platform_tags:
        if tag == "any":
            platform_match = True
            break
        if tag.startswith(target_os_prefix):
            # Basic architecture check (amd64 vs arm64)
            if "amd64" in tag or "x86_64" in tag:
                # We assume x64 for now as it's standard for ComfyUI
                platform_match = True
                break
            if "arm64" in tag or "aarch64" in tag:
                 # TODO: Add arm64 audit detection if needed
                 platform_match = True
                 break

    if not platform_match:
        return False, f"Platform mismatch: Wheel is for {', '.join(metadata.platform_tags)}, but OS is {audit.os}"

    # 2. Python Version Check
    # Tags like 'py3', 'cp310', 'py310', 'cp38'
    python_match = False
    target_ver_clean = audit.python_target_version.replace(".", "") # e.g. "310"
    
    for tag in metadata.python_tags:
        if tag == "py3" or tag == "py2.py3":
            python_match = True
            break
        # Match 'cp310' or 'py310'
        if target_ver_clean in tag:
            python_match = True
            break
        # Logic for 'cp3' usually means any python 3.x
        if tag == "cp3" or tag == "py3":
             python_match = True
             break

    if not python_match:
        return False, f"Python version mismatch: Wheel is for {', '.join(metadata.python_tags)}, but environment is {audit.python_target_version}"

    return True, None
