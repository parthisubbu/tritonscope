"""JSON output formatter."""

import json
from pathlib import Path
from typing import Any, Dict

from tritonscope.pipeline import AnalysisReport


class JSONFormatter:
    """Format analysis report as JSON."""

    @staticmethod
    def format(report: AnalysisReport) -> str:
        """Format report as JSON string.

        Returns:
            JSON string representation of report
        """
        data = JSONFormatter.to_dict(report)
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def to_dict(report: AnalysisReport) -> Dict[str, Any]:
        """Convert report to dict."""
        return {
            "kernel_name": report.kernel_name,
            "config": {
                "constexpr_values": report.config.constexpr_values,
                "num_warps": report.config.num_warps,
                "num_stages": report.config.num_stages,
            },
            "summary": {
                "total_diagnostics": len(report.all_diagnostics),
                "errors": report.error_count,
                "warnings": report.warning_count,
                "info": report.info_count,
            },
            "diagnostics": [
                {
                    "id": d.id,
                    "category": d.category,
                    "severity": d.severity.value,
                    "basis": d.basis.value,
                    "message": d.message,
                    "evidence": d.evidence,
                    "suggestion": d.suggestion,
                    "location": {
                        "file": str(d.location.file) if d.location else None,
                        "line": d.location.line if d.location else None,
                        "function": d.location.function if d.location else None,
                    } if d.location else None,
                }
                for d in report.all_diagnostics
            ],
            "analyzers": [
                {
                    "name": result.analyzer_name,
                    "passed": result.passed,
                    "diagnostics_count": len(result.diagnostics),
                    "errors": result.errors,
                }
                for result in report.analyzer_results
            ],
        }

    @staticmethod
    def write_file(report: AnalysisReport, path: Path) -> None:
        """Write report to JSON file.

        Args:
            report: Analysis report
            path: File path to write to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(JSONFormatter.format(report))
