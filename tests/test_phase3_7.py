"""Comprehensive tests for Phases 3-7."""

import json
import tempfile
from pathlib import Path

import pytest

from tritonscope.analyzers.base import Analyzer, Diagnostic, Severity, Basis
from tritonscope.analyzers.ast_analyzer import ASTAnalyzer
from tritonscope.analyzers.runtime_analyzer import RuntimeAnalyzer
from tritonscope.core.launcher import KernelConfig
from tritonscope.core.loader import KernelLoader, KernelMetadata, KernelParameter
from tritonscope.output.json_formatter import JSONFormatter
from tritonscope.output.terminal_formatter import TerminalFormatter
from tritonscope.pipeline import AnalysisPipeline, AnalysisReport


class TestAnalyzers:
    """Test analyzer implementations."""

    @pytest.fixture
    def kernel_metadata(self):
        """Create test kernel metadata."""
        return KernelMetadata(
            name="test_kernel",
            file_path=Path("test.py"),
            line_number=1,
            parameters=[
                KernelParameter("X", is_constexpr=False),
                KernelParameter("Y", is_constexpr=False),
                KernelParameter("BLOCK_SIZE", is_constexpr=True),
            ],
            source_code="@triton.jit\ndef test_kernel(X, Y, BLOCK_SIZE: tl.constexpr):\n    pass",
        )

    @pytest.fixture
    def kernel_config(self):
        """Create test kernel config."""
        return KernelConfig(
            kernel_name="test_kernel",
            constexpr_values={"BLOCK_SIZE": 256},
            num_warps=4,
            num_stages=3,
        )

    def test_ast_analyzer(self, kernel_metadata, kernel_config):
        """Test AST analyzer."""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(kernel_metadata, kernel_config)

        assert result.analyzer_name == "AST"
        assert isinstance(result.diagnostics, list)

    def test_runtime_analyzer(self, kernel_metadata, kernel_config):
        """Test runtime analyzer."""
        analyzer = RuntimeAnalyzer()
        result = analyzer.analyze(kernel_metadata, kernel_config)

        assert result.analyzer_name == "Runtime"
        assert len(result.diagnostics) >= 0

    def test_analyzer_isolation(self, kernel_metadata, kernel_config):
        """Test analyzer isolation (safe_analyze)."""
        analyzer = ASTAnalyzer()
        # Safe analyze should never throw
        result = analyzer._safe_analyze(kernel_metadata, kernel_config)

        assert result.analyzer_name == "AST"
        assert isinstance(result.passed, bool)


class TestDiagnostics:
    """Test diagnostic objects."""

    def test_diagnostic_creation(self):
        """Test creating a diagnostic."""
        diag = Diagnostic(
            id="TS001",
            category="Occupancy",
            severity=Severity.WARNING,
            basis=Basis.COMPUTED,
            message="Low occupancy",
            evidence="25% occupancy",
            suggestion="Reduce BLOCK_SIZE",
        )

        assert diag.id == "TS001"
        assert diag.severity == Severity.WARNING
        assert str(diag)  # Should have readable string repr

    def test_diagnostic_severity_enum(self):
        """Test severity levels."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"


class TestPipeline:
    """Test analysis pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create analysis pipeline."""
        return AnalysisPipeline()

    @pytest.fixture
    def kernel_setup(self):
        """Create kernel and config."""
        metadata = KernelMetadata(
            name="softmax",
            file_path=Path("softmax.py"),
            line_number=10,
            parameters=[
                KernelParameter("X", is_constexpr=False),
                KernelParameter("Y", is_constexpr=False),
                KernelParameter("BLOCK_SIZE", is_constexpr=True),
            ],
            source_code="@triton.jit\ndef softmax(X, Y, BLOCK_SIZE: tl.constexpr):\n    pass",
        )
        config = KernelConfig(
            kernel_name="softmax",
            constexpr_values={"BLOCK_SIZE": 256},
        )
        return metadata, config

    def test_pipeline_analyze(self, pipeline, kernel_setup):
        """Test pipeline analysis."""
        metadata, config = kernel_setup
        report = pipeline.analyze(metadata, config)

        assert isinstance(report, AnalysisReport)
        assert report.kernel_name == "softmax"
        assert len(report.analyzer_results) > 0
        assert all(isinstance(r, type(report.analyzer_results[0])) for r in report.analyzer_results)

    def test_pipeline_report_summary(self, pipeline, kernel_setup):
        """Test report summary."""
        metadata, config = kernel_setup
        report = pipeline.analyze(metadata, config)

        assert report.error_count >= 0
        assert report.warning_count >= 0
        assert report.info_count >= 0
        summary = report.summary()
        assert isinstance(summary, str)


class TestOutputFormatters:
    """Test output formatters."""

    @pytest.fixture
    def test_report(self):
        """Create test analysis report."""
        metadata = KernelMetadata(
            name="test",
            file_path=Path("test.py"),
            line_number=1,
            parameters=[],
        )
        config = KernelConfig(
            kernel_name="test",
            constexpr_values={"BLOCK_SIZE": 256},
        )
        report = AnalysisReport(kernel_name="test", config=config)

        # Add a diagnostic
        diag = Diagnostic(
            id="TS001",
            category="Test",
            severity=Severity.WARNING,
            basis=Basis.HEURISTIC,
            message="Test diagnostic",
            evidence="Test evidence",
            suggestion="Test suggestion",
        )
        report.analyzer_results.append(
            type('Result', (), {
                'analyzer_name': 'TestAnalyzer',
                'passed': True,
                'diagnostics': [diag],
                'errors': []
            })()
        )
        return report

    def test_json_formatter(self, test_report):
        """Test JSON formatter."""
        json_str = JSONFormatter.format(test_report)
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["kernel_name"] == "test"
        assert "diagnostics" in data

    def test_json_formatter_write_file(self, test_report):
        """Test JSON formatter file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            JSONFormatter.write_file(test_report, path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["kernel_name"] == "test"

    def test_terminal_formatter(self, test_report):
        """Test terminal formatter."""
        formatter = TerminalFormatter()
        # Should not raise
        formatter.format(test_report)


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_analysis(self):
        """Test end-to-end analysis flow."""
        # Create a kernel file
        code = '''
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    """Softmax kernel."""
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            kernel_file = Path(f.name)

        try:
            # Load kernel
            loader = KernelLoader(kernel_file)
            kernel = loader.find_kernel("softmax_kernel")
            assert kernel is not None

            # Generate config
            config = KernelConfig(
                kernel_name="softmax_kernel",
                constexpr_values={"BLOCK_SIZE": 256},
            )

            # Run analysis
            pipeline = AnalysisPipeline()
            report = pipeline.analyze(kernel, config)

            # Format output
            json_output = JSONFormatter.format(report)
            assert isinstance(json_output, str)

            # Should be valid JSON
            data = json.loads(json_output)
            assert data["kernel_name"] == "softmax_kernel"

        finally:
            kernel_file.unlink()
