"""Analysis pipeline orchestration."""

from dataclasses import dataclass, field
from typing import Any, List

from tritonscope.analyzers.base import AnalyzerResult, Diagnostic, Severity
from tritonscope.analyzers.ast_analyzer import ASTAnalyzer
from tritonscope.analyzers.runtime_analyzer import RuntimeAnalyzer
from tritonscope.core.loader import KernelMetadata
from tritonscope.core.launcher import KernelConfig


@dataclass
class AnalysisReport:
    """Complete analysis report for a kernel."""

    kernel_name: str
    config: KernelConfig
    analyzer_results: List[AnalyzerResult] = field(default_factory=list)

    @property
    def all_diagnostics(self) -> List[Diagnostic]:
        """Get all diagnostics from all analyzers."""
        diagnostics = []
        for result in self.analyzer_results:
            diagnostics.extend(result.diagnostics)
        return sorted(diagnostics, key=lambda d: (d.severity.value, d.id))

    @property
    def error_count(self) -> int:
        """Count of error-level diagnostics."""
        return sum(1 for d in self.all_diagnostics if d.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level diagnostics."""
        return sum(1 for d in self.all_diagnostics if d.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of info-level diagnostics."""
        return sum(1 for d in self.all_diagnostics if d.severity == Severity.INFO)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Kernel: {self.kernel_name}\n"
            f"Config: {self.config.constexpr_values}\n"
            f"Diagnostics: {self.error_count} errors, {self.warning_count} warnings, {self.info_count} info"
        )


class AnalysisPipeline:
    """Orchestrate kernel analysis."""

    def __init__(self):
        self.analyzers = [
            ASTAnalyzer(),
            RuntimeAnalyzer(),
        ]

    def analyze(
        self,
        kernel_metadata: KernelMetadata,
        config: KernelConfig,
    ) -> AnalysisReport:
        """Run full analysis pipeline.

        Args:
            kernel_metadata: Kernel metadata from loader
            config: Launch configuration

        Returns:
            AnalysisReport with all diagnostics
        """
        report = AnalysisReport(
            kernel_name=kernel_metadata.name,
            config=config,
        )

        # Run each analyzer independently
        for analyzer in self.analyzers:
            result = analyzer._safe_analyze(kernel_metadata, config)
            report.analyzer_results.append(result)

        return report
