"""Terminal output formatter with rich colors."""

from rich.console import Console
from rich.table import Table

from tritonscope.pipeline import AnalysisReport


class TerminalFormatter:
    """Format analysis report for terminal output."""

    def __init__(self, console: Console = None):
        self.console = console or Console()

    def format(self, report: AnalysisReport) -> None:
        """Print formatted report to terminal.

        Args:
            report: Analysis report
        """
        self._print_header(report)
        self._print_summary(report)
        self._print_diagnostics(report)
        self._print_analyzer_status(report)

    def _print_header(self, report: AnalysisReport) -> None:
        """Print header with kernel name and config."""
        self.console.print(f"\n[bold magenta]TrionScope Analysis Report[/bold magenta]")
        self.console.print(f"[cyan]Kernel:[/cyan] {report.kernel_name}")
        self.console.print(f"[cyan]Config:[/cyan] {report.config.constexpr_values}\n")

    def _print_summary(self, report: AnalysisReport) -> None:
        """Print summary statistics."""
        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        table.add_row("Total Diagnostics", str(len(report.all_diagnostics)))
        table.add_row("[red]Errors[/red]", str(report.error_count))
        table.add_row("[yellow]Warnings[/yellow]", str(report.warning_count))
        table.add_row("[blue]Info[/blue]", str(report.info_count))

        self.console.print(table)
        self.console.print()

    def _print_diagnostics(self, report: AnalysisReport) -> None:
        """Print diagnostics table."""
        if not report.all_diagnostics:
            self.console.print("[green]✓ No issues found![/green]\n")
            return

        table = Table(title="Diagnostics", box=None)
        table.add_column("ID", style="cyan")
        table.add_column("Category")
        table.add_column("Message")
        table.add_column("Severity")

        for diag in report.all_diagnostics:
            severity_color = {
                "error": "[red]ERROR[/red]",
                "warning": "[yellow]WARN[/yellow]",
                "info": "[blue]INFO[/blue]",
            }.get(diag.severity.value, "INFO")

            table.add_row(
                diag.id,
                diag.category,
                diag.message,
                severity_color,
            )

        self.console.print(table)
        self.console.print()

        # Print details for each diagnostic
        for diag in report.all_diagnostics:
            self.console.print(f"[bold]{diag.id}:[/bold] {diag.message}")
            self.console.print(f"  [dim]Evidence:[/dim] {diag.evidence}")
            self.console.print(f"  [dim]Suggestion:[/dim] {diag.suggestion}\n")

    def _print_analyzer_status(self, report: AnalysisReport) -> None:
        """Print analyzer run status."""
        table = Table(title="Analyzers", show_header=False, box=None)
        table.add_column("Analyzer", style="cyan")
        table.add_column("Status")

        for result in report.analyzer_results:
            status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            table.add_row(result.analyzer_name, status)

        self.console.print(table)
