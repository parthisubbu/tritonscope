"""Markdown output formatter."""

from tritonscope.pipeline import AnalysisReport


class MarkdownFormatter:
    """Format analysis report as Markdown."""

    @staticmethod
    def format(report: AnalysisReport) -> str:
        """Format report as Markdown.

        Returns:
            Markdown string
        """
        lines = [
            f"# TrionScope Analysis Report",
            f"",
            f"**Kernel:** `{report.kernel_name}`",
            f"**Config:** `{report.config.constexpr_values}`",
            f"",
            f"## Summary",
            f"",
            f"- **Total Diagnostics:** {len(report.all_diagnostics)}",
            f"- **Errors:** {report.error_count}",
            f"- **Warnings:** {report.warning_count}",
            f"- **Info:** {report.info_count}",
            f"",
            f"## Diagnostics",
            f"",
        ]

        if not report.all_diagnostics:
            lines.append("✓ No issues found!")
        else:
            for diag in report.all_diagnostics:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(diag.severity.value, "•")
                lines.extend([
                    f"### {severity_icon} {diag.id}: {diag.message}",
                    f"",
                    f"**Category:** {diag.category}",
                    f"**Basis:** {diag.basis.value}",
                    f"**Evidence:** {diag.evidence}",
                    f"**Suggestion:** {diag.suggestion}",
                    f"",
                ])

        return "\n".join(lines)
