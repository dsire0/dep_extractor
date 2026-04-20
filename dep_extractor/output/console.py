"""
dep_extractor/output/console.py
================================
All terminal output for the dep_extractor package, powered by `rich`.

Educational: The original monolith scattered 50+ raw print() calls throughout
the logic code. This violated the separation of concerns principle — output
formatting was interleaved with business logic, making it impossible to run
the pipeline in non-interactive or JSON-output modes.

This module:
  1. Provides a single shared console = Console() instance
  2. Encodes ADHD-forward visual design (per ADHD Focus Helper KI):
     - Phase headers use bordered panels with emoji anchors
     - Conflicts shown in color-coded rich.Table
     - URL validation shown as a live Progress bar (not a wall of text)
  3. Is the ONLY module that imports from `rich` — no other module calls print()

Usage:
    from dep_extractor.output.console import console, print_phase_header
    print_phase_header(1, "Scanning Nodes", "🔍")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

if TYPE_CHECKING:
    from dep_extractor.models import (
        AuditData,
        ConflictReport,
        ScanResult,
        UrlValidationResult,
    )


# ---------------------------------------------------------------------------
# Shared console instance
# ---------------------------------------------------------------------------

if _RICH_AVAILABLE:
    console = Console(highlight=True, markup=True)
else:
    # Minimal shim so imports never fail even without rich installed
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, title="", **kwargs):
            print(f"\n{'='*60}\n{title}\n{'='*60}")

    console = _FallbackConsole()


def _check_rich(func_name: str) -> bool:
    if not _RICH_AVAILABLE:
        console.print(f"[{func_name}] rich not installed — using plain text output.")
        return False
    return True


# ---------------------------------------------------------------------------
# Phase UI
# ---------------------------------------------------------------------------

def print_phase_header(phase_num: int, title: str, emoji: str = "⚙️") -> None:
    """
    Print an ADHD-anchored phase header as a rich Panel.

    Educational: Panels create a strong visual boundary that helps users
    track where they are in a multi-phase pipeline without reading every
    line of output. The emoji provides a quick scannable marker.

    Example output:
    ┌─────────────────────────────────────────────────────┐
    │  🔍 Phase 1: Scanning Nodes                         │
    └─────────────────────────────────────────────────────┘
    """
    if not _check_rich("print_phase_header"):
        print(f"\n{'─'*55}\n{emoji} Phase {phase_num}: {title}\n{'─'*55}")
        return

    console.print()
    console.print(
        Panel(
            f"[bold white]{emoji}  Phase {phase_num}: {title}[/bold white]",
            border_style="bright_blue",
            expand=True,
        )
    )


def print_section(title: str, emoji: str = "›") -> None:
    """Print a lightweight sub-section rule (no panel box)."""
    if _RICH_AVAILABLE:
        console.rule(f"[bold cyan]{emoji} {title}[/bold cyan]", style="dim cyan")
    else:
        print(f"\n{emoji} {title}")


# ---------------------------------------------------------------------------
# Scan output
# ---------------------------------------------------------------------------

def print_scan_summary(result: "ScanResult") -> None:
    """Displays a compact summary of the scan phase results."""
    nodes = len({r.source_node for r in result.packages})

    if not _RICH_AVAILABLE:
        print(
            f"Scan: {len(result.req_files)} files, "
            f"{len(result.packages)} packages, "
            f"{len(result.specialized)} specialized, "
            f"from {nodes} nodes."
        )
        return

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bright_cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Nodes scanned", str(nodes))
    table.add_row("Requirements files", str(len(result.req_files)))
    table.add_row("Standard packages", str(len(result.packages)))
    table.add_row("Specialized (URL/path)", str(len(result.specialized)))

    console.print(table)


# ---------------------------------------------------------------------------
# Audit output
# ---------------------------------------------------------------------------

def print_audit_table(audit: "AuditData") -> None:
    """Displays hardware audit results in a formatted rich Table."""
    if not _RICH_AVAILABLE:
        print(f"  OS: {audit.os} | CUDA: {audit.cuda_version} | uv: {audit.has_uv}")
        return

    table = Table(
        title="[bold]🖥️  System Audit[/bold]",
        box=box.ROUNDED,
        border_style="blue",
        show_lines=False,
    )
    table.add_column("Property", style="bright_cyan")
    table.add_column("Value", style="white")

    table.add_row("Operating System", f"{audit.os} {audit.os_release}")
    table.add_row("Windows Insider", "✅ Yes" if audit.is_insider else "No")
    table.add_row("GINGER-class GPU", "✅ 4090/5090 detected" if audit.is_ginger_class else "No")
    table.add_row("uv available", "✅ Yes" if audit.has_uv else "❌ No")

    if audit.gpus:
        gpu_str = " | ".join(f"{g.name} ({g.vram_gb}GB)" for g in audit.gpus)
        table.add_row("GPU(s)", gpu_str)
        table.add_row("Total VRAM", f"{audit.total_vram_gb} GB")

    table.add_row("CUDA Version", audit.cuda_version or "Not detected")
    table.add_row("Python target", audit.python_target_version)
    table.add_row("Terminal", audit.terminal)

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Conflict output
# ---------------------------------------------------------------------------

def print_conflict_tree(conflicts: list["ConflictReport"]) -> None:
    """
    Displays version conflicts in a color-coded rich.Table.

    ADHD design: Severity is immediately visible via color — red = hard block,
    yellow = soft warning. Each row shows the package, the conflicting specs,
    and which nodes are involved.
    """
    if not conflicts:
        if _RICH_AVAILABLE:
            console.print("[bold green]✅ No version conflicts detected.[/bold green]")
        else:
            print("✅ No version conflicts detected.")
        return

    if not _RICH_AVAILABLE:
        for c in conflicts:
            print(str(c))
        return

    table = Table(
        title=f"[bold red]⚠️  {len(conflicts)} Version Conflict(s) Detected[/bold red]",
        box=box.ROUNDED,
        border_style="red",
        show_lines=True,
    )
    table.add_column("Severity", style="bold", no_wrap=True, width=8)
    table.add_column("Package", style="bright_white")
    table.add_column("Specs", style="yellow")
    table.add_column("Nodes", style="dim white")

    for c in conflicts:
        sev_style = "bold red" if c.severity == "hard" else "bold yellow"
        table.add_row(
            Text(c.severity.upper(), style=sev_style),
            c.package,
            " vs ".join(c.specs),
            ", ".join(n.get("node", "?") for n in c.nodes[:4]),
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# URL Validation output
# ---------------------------------------------------------------------------

def make_url_validation_progress() -> "Progress":
    """
    Returns a configured rich Progress bar for live URL validation display.

    Usage (from cli.py):
        with make_url_validation_progress() as progress:
            task = progress.add_task("Validating URLs", total=len(urls))
            for result in validate_one_by_one(urls):
                progress.advance(task)
    """
    if not _RICH_AVAILABLE:
        raise RuntimeError("rich is required for progress bars")

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_url_validation_results(results: list["UrlValidationResult"]) -> None:
    """Prints URL validation results as a compact table."""
    if not results:
        return

    if not _RICH_AVAILABLE:
        for r in results:
            status = "OK" if r.is_valid else f"FAIL: {r.error}"
            print(f"  [{status}] {r.url}")
        return

    table = Table(
        title="[bold]🌐 URL Validation Results[/bold]",
        box=box.SIMPLE,
        border_style="cyan",
    )
    table.add_column("Status", width=7, no_wrap=True)
    table.add_column("Latency", width=8, justify="right")
    table.add_column("URL", style="dim white", overflow="fold")

    for r in results:
        if r.is_valid:
            status = Text("✅ OK", style="bold green")
            latency = f"{r.latency_ms:.0f}ms"
        else:
            status = Text("❌ FAIL", style="bold red")
            latency = "-"

        table.add_row(status, latency, r.url[:80])

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Installation output
# ---------------------------------------------------------------------------

def print_installation_summary(
    base: list[str],
    heavy: list[str],
    skipped: list[str],
) -> None:
    """Print a compact installation phase summary."""
    if not _RICH_AVAILABLE:
        print(f"\nInstall summary: {len(base)} base, {len(heavy)} heavy, {len(skipped)} skipped.")
        return

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bright_cyan")
    table.add_column("Value", style="white")

    table.add_row("To install (base)", str(len(base)))
    table.add_row("To build (heavy)", str(len(heavy)))
    table.add_row("Already installed", f"[dim]{len(skipped)} skipped[/dim]")

    console.print()
    console.print(table)


def print_success(message: str) -> None:
    """Print a success message."""
    if _RICH_AVAILABLE:
        console.print(f"[bold green]✅ {message}[/bold green]")
    else:
        print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    if _RICH_AVAILABLE:
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
    else:
        print(f"⚠️  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    if _RICH_AVAILABLE:
        console.print(f"[bold red]❌ {message}[/bold red]")
    else:
        print(f"❌ {message}")
