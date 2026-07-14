"""TrionScope CLI - Command-line interface for kernel analysis."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tritonscope.__version__ import __version__
from tritonscope.compatibility import run_compatibility_check
from tritonscope.hardware import get_system_info

app = typer.Typer(help="Diagnose Triton GPU kernel performance issues")
console = Console()


@app.command()
def analyze(
    kernel_file: Path = typer.Argument(..., help="Python file containing @triton.jit kernel"),
    kernel: Optional[str] = typer.Option(None, help="Kernel function name to analyze"),
    config: Optional[str] = typer.Option(None, help="Kernel config (e.g., 'BLOCK_SIZE=256')"),
    output: Optional[Path] = typer.Option(None, help="Output directory for analysis bundle"),
    json: Optional[Path] = typer.Option(None, help="Write JSON report to file"),
    markdown: Optional[Path] = typer.Option(None, help="Write Markdown report to file"),
    html: Optional[Path] = typer.Option(None, help="Write HTML report to file"),
    fail_on: str = typer.Option("never", help="Exit code 2 if severity >= {high,medium,low,never}"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
) -> None:
    """Analyze a Triton kernel for performance issues.

    Example:
        tritonscope analyze my_softmax.py --kernel softmax --config "BLOCK_SIZE=512"
    """
    # Phase 0: Just verify the file exists
    if not kernel_file.exists():
        console.print(f"[red]✗ File not found: {kernel_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]→ Analyzing {kernel_file}[/blue]")
    if verbose:
        console.print(f"  Kernel: {kernel}")
        console.print(f"  Config: {config}")
        console.print(f"  Output: {output}")

    # Phase 2+: Full implementation
    console.print("[yellow]! Phase 0: Skeleton only. Full analysis in Phase 2+[/yellow]")
    raise typer.Exit(0)


@app.command()
def doctor() -> None:
    """Run system diagnostics for TrionScope environment.

    Comprehensive checks:
      - Python version (3.10+)
      - CUDA & GPU availability
      - PyTorch & Triton versions
      - C++ compiler (for Triton compilation)
      - Cache directory permissions
    """
    console.print("[bold magenta]╔════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║     TrionScope System Diagnostics       ║[/bold magenta]")
    console.print("[bold magenta]╚════════════════════════════════════════╝[/bold magenta]\n")

    console.print(f"[cyan]TrionScope {__version__}[/cyan]\n")

    # Run compatibility checks
    report = run_compatibility_check()

    # Display checks in table
    table = Table(title="Compatibility Checks", show_header=False, box=None)
    table.add_column("Status", width=2, style="bold")
    table.add_column("Check", width=30)
    table.add_column("Details", style="dim")

    for check in report.checks:
        status_icon = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
        details = check.message or (f"v{check.actual}" if check.actual else "")
        table.add_row(status_icon, check.name, details)

    console.print(table)

    # System info summary
    console.print("\n[bold]System Information[/bold]")
    sys_info = get_system_info()

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", width=20, style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Python", sys_info["python"])
    info_table.add_row("Platform", sys_info["platform"].split("-")[0])

    if sys_info["gpu"].available:
        info_table.add_row("[green]GPU[/green]", f"{sys_info['gpu'].name} (cc {sys_info['gpu'].compute_capability})")
        info_table.add_row("Memory", f"{sys_info['gpu'].memory_gb:.1f} GB")
    else:
        info_table.add_row("GPU", "[yellow]Not available (CPU-only mode)[/yellow]")

    if sys_info["pytorch"]["installed"]:
        info_table.add_row("PyTorch", sys_info["pytorch"]["version"])
    if sys_info["triton"]["installed"]:
        info_table.add_row("Triton", sys_info["triton"]["version"])

    console.print(info_table)

    # Summary
    console.print()
    if report.passed:
        console.print("[green bold]✓ All checks passed! TrionScope is ready to use.[/green bold]")
    else:
        if report.failures:
            console.print(f"[red bold]✗ {len(report.failures)} critical issue(s):[/red bold]")
            for failure in report.failures:
                console.print(f"  [red]•[/red] {failure}")
        if report.warnings:
            console.print(f"[yellow bold]⚠ {len(report.warnings)} warning(s):[/yellow bold]")
            for warning in report.warnings:
                console.print(f"  [yellow]•[/yellow] {warning}")

    console.print("\nFor GPU analysis, install: [cyan]pip install tritonscope[gpu][/cyan]")


@app.command()
def version() -> None:
    """Show TrionScope version."""
    console.print(f"TrionScope {__version__}")


@app.callback()
def main(ctx: typer.Context) -> None:
    """TrionScope - Diagnose Triton GPU kernel performance in seconds."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
