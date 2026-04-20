"""
dep_extractor/cli.py
======================
Command-line entry point — thin wrapper only.

Educational: The CLI layer has ONE job: parse arguments, wire together the
pipeline modules, and translate failures into exit codes. No business logic
lives here. This separation means:
  1. The entire pipeline is testable without invoking argparse
  2. A future GUI or API layer can reuse all pipeline modules directly
  3. sys.exit() is called in EXACTLY ONE PLACE (this file), not inside library code

Flag compatibility: 100% backward-compatible with the original
requirements_extractor.py CLI. All existing .bat launchers continue to work
with zero modifications.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Return a configured argparse parser.

    All flags are 100% backward-compatible with requirements_extractor.py.
    """
    parser = argparse.ArgumentParser(
        prog="dep_extractor",
        description=(
            "ComfyUI Dependency Resolver Suite — scans custom_nodes, "
            "detects conflicts, validates URLs, and installs packages."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        metavar="DIR",
        help="Root directory to scan for custom node requirement files (default: current dir)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to target Python interpreter (e.g. path/to/venv/Scripts/python.exe)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_requirements.txt",
        metavar="FILE",
        help="Output file path for combined requirements (default: combined_requirements.txt)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip URL validation (faster for offline environments)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run UV SAT-solver dry-run to validate dependency resolution",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Display the hardware audit table (OS, GPU, CUDA, uv)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Generate report only — skip the installation phase",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _setup_environment() -> None:
    """
    Apply environment stabilisation constants at process startup.

    Educational: Setting these once here (not scattered across the codebase)
    ensures they are always applied before any other code runs, regardless
    of import order.
    """
    from dep_extractor.config import ENV_STABILISATION
    for key, value in ENV_STABILISATION.items():
        os.environ.setdefault(key, value)

    # Force UTF-8 terminal output on Windows to avoid UnicodeEncodeError
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _setup_logging(debug: bool) -> None:
    """Configure root logging level based on --debug flag."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full pipeline orchestration:
        1. Scan  — discover all requirements.txt files
        2. Audit — probe hardware environment
        3. Conflict — static pre-resolution checks
        4. Simulate — UV SAT-solver dry-run (if --simulate)
        5. Validate — concurrent URL validation (unless --quick)
        6. Write  — generate combined_requirements.txt
        7. Install — phased installation (unless --no-install)
    """
    _setup_environment()
    args = build_parser().parse_args()
    _setup_logging(args.debug)

    # Import pipeline modules after environment is stabilised
    from dep_extractor.analysis.conflict_detector import (
        detect_static_conflicts,
        format_conflict_report_text,
    )
    from dep_extractor.analysis.url_validator import validate_urls_sync
    from dep_extractor.analysis.wheel_tracker import classify_wheels, find_workspace_wheels
    from dep_extractor.core.auditor import (
        get_installed_packages,
        run_hardware_audit,
        suggest_best_indices,
    )
    from dep_extractor.core.normalizer import RequirementNormalizer
    from dep_extractor.core.scanner import scan_nodes
    from dep_extractor.core.simulator import simulate_resolution
    from dep_extractor.install.installer import execute_phased_installation
    from dep_extractor.models import AuditReport
    from dep_extractor.output.console import (
        console,
        print_audit_table,
        print_conflict_tree,
        print_phase_header,
        print_scan_summary,
        print_url_validation_results,
    )
    from dep_extractor.output.report_writer import write_combined_requirements

    root = Path(args.root).resolve()
    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Phase 1: Scan
    # ------------------------------------------------------------------
    print_phase_header(1, "Scanning Custom Nodes", "🔍")
    normalizer = RequirementNormalizer()
    scan = scan_nodes(root, normalizer)
    print_scan_summary(scan)

    # ------------------------------------------------------------------
    # Phase 2: Hardware Audit
    # ------------------------------------------------------------------
    print_phase_header(2, "Hardware Audit", "◇")
    audit = run_hardware_audit(python_executable=args.python)
    installed = get_installed_packages(python_executable=audit.python_executable)

    # Always show audit results (Phase 2 core requirement)
    print_audit_table(audit)

    # ------------------------------------------------------------------
    # Phase 2.5: Wheel Compatibility Audit
    # ------------------------------------------------------------------
    incompatible_wheels = []
    for req in scan.specialized:
        if req.is_wheel and req.wheel_metadata:
            from dep_extractor.utils.wheel_parser import check_wheel_compatibility
            is_comp, reason = check_wheel_compatibility(req.wheel_metadata, audit)
            req.wheel_metadata.is_compatible = is_comp
            req.wheel_metadata.compatibility_reason = reason
            if not is_comp:
                incompatible_wheels.append((req, reason))

    if incompatible_wheels:
        from dep_extractor.output.console import print_section, print_warning
        print_section("Wheel Compatibility Check", "🛞")
        for req, reason in incompatible_wheels:
            print_warning(f"Incompatible wheel found: [white]{req.name}[/white]\n    {reason}")
        console.print()

    # ------------------------------------------------------------------
    # Phase 3: Static Conflict Detection
    # ------------------------------------------------------------------
    print_phase_header(3, "Static Conflict Analysis", "⚡")
    static_conflicts = detect_static_conflicts(scan.origin_map)
    print_conflict_tree(static_conflicts)

    # ------------------------------------------------------------------
    # Phase 3.5: SAT Simulation (optional)
    # ------------------------------------------------------------------
    extra_indices = suggest_best_indices(audit)
    if args.simulate:
        print_phase_header(4, "SAT Solver Validation", "◌")
        if not audit.has_uv:
            console.print("[yellow]uv not found — skipping simulation.[/yellow]")
        else:
            with console.status("[bold cyan]Resolving global SAT constraints with UV...[/bold cyan]", spinner="dots"):
                sim = simulate_resolution(
                    req_files=scan.req_files,
                    python_version=audit.python_target_version,
                    extra_indices=extra_indices,
                    debug=args.debug,
                )
            if sim.success:
                console.print(
                    f"[bold green]✅ Resolution succeeded — "
                    f"{len(sim.solved_map)} packages solved.[/bold green]"
                )
                audit.solved_map = sim.solved_map
            else:
                if not sim.conflicts:
                    console.print("[bold red]UV SAT solver failed with unknown error:[/bold red]")
                    from rich.panel import Panel
                    console.print(Panel(sim.raw_output.strip(), title="Raw UV Output", border_style="red"))
                else:
                    print_conflict_tree(sim.conflicts)
                
                console.print()
                console.print(
                    "[bold red]SAT resolution failed. Fix conflicts above before installing.[/bold red]"
                )
                sys.exit(1)    # ← Only sys.exit in the entire codebase

    # ------------------------------------------------------------------
    # Phase 4: URL Validation (skipped with --quick)
    # ------------------------------------------------------------------
    url_results: list = []
    if not args.quick and scan.online_urls:
        print_phase_header(5, "URL Validation", "⊕")
        console.print(f"   Validating [cyan]{len(scan.online_urls)}[/cyan] URLs concurrently...")
        url_results = validate_urls_sync(scan.online_urls)
        print_url_validation_results(url_results)

    # ------------------------------------------------------------------
    # Wheel Tracking
    # ------------------------------------------------------------------
    wheels = find_workspace_wheels(root)
    _tracked, untracked = classify_wheels(wheels, scan.specialized, root)
    untracked_names = [w.name for w in untracked]

    # ------------------------------------------------------------------
    # Phase 4: Write Output
    # ------------------------------------------------------------------
    print_phase_header(6, "Writing Combined Requirements", "▣")
    audit_report = AuditReport(
        untracked_wheels=untracked_names,
        conflict_causality=static_conflicts,
    )
    write_combined_requirements(
        output_path=output_path,
        packages=scan.packages,
        specialized=scan.specialized,
        extra_indices=extra_indices,
        audit=audit,
        audit_report=audit_report,
        url_results=url_results,
        root=root,
    )
    console.print(f"   [bold green]✅ Written:[/bold green] {output_path}")

    # ------------------------------------------------------------------
    # Phase 5 & 6: Installation (skipped with --no-install)
    # ------------------------------------------------------------------
    if not args.no_install:
        summary = execute_phased_installation(audit, scan, installed)
        if not summary.success:
            console.print(
                f"[bold yellow]⚠️  Installation completed with "
                f"{len(summary.failed)} failure(s).[/bold yellow]"
            )

    console.print()
    console.print("[bold green]✅ dep_extractor run complete.[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
