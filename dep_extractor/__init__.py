"""
dep_extractor — ComfyUI Dependency Resolver Suite (v2.0)

Modular, testable, concurrent replacement for requirements_extractor.py.

Public API surface (everything needed to use this package programmatically):
    from dep_extractor.core.scanner import scan_nodes
    from dep_extractor.core.auditor import run_hardware_audit
    from dep_extractor.core.simulator import simulate_resolution
    from dep_extractor.analysis.conflict_detector import detect_static_conflicts
    from dep_extractor.analysis.url_validator import validate_urls_sync
    from dep_extractor.output.report_writer import write_combined_requirements
    from dep_extractor.install.installer import execute_phased_installation
"""

__version__ = "2.0.0"
__author__ = "thegingeroriginator"
__description__ = "ComfyUI Dependency Resolver Suite — streamlined package"
