"""Base analyzer classes for TrionScope."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Diagnostic severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Basis(str, Enum):
    """Evidence basis for a diagnostic."""
    ARTIFACT = "artifact"  # Observed in compiled output
    COMPUTED = "computed"  # From deterministic math
    HEURISTIC = "heuristic"  # Rule of thumb


@dataclass
class SourceLocation:
    """Source code location."""
    file: Path
    line: int
    column: int = 0
    function: Optional[str] = None


@dataclass
class Diagnostic:
    """A single diagnostic finding."""

    id: str  # TS001, TS002, etc.
    category: str  # Occupancy, Memory, Registers, etc.
    severity: Severity
    basis: Basis
    message: str
    evidence: str  # Why this fired
    suggestion: str  # How to fix
    location: Optional[SourceLocation] = None

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.severity.value, "•")
        return f"{icon} [{self.id}] {self.message}"


@dataclass
class AnalyzerResult:
    """Result from running an analyzer."""

    analyzer_name: str
    passed: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_diagnostic(self, diag: Diagnostic) -> None:
        """Add a diagnostic."""
        self.diagnostics.append(diag)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def summary(self) -> str:
        """Human-readable summary."""
        if self.passed:
            return f"✓ {self.analyzer_name}: OK"
        return f"✗ {self.analyzer_name}: {len(self.errors)} error(s), {len(self.diagnostics)} diagnostic(s)"


class Analyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"tritonscope.{name}")

    @abstractmethod
    def analyze(self, kernel_metadata: Any, config: Any) -> AnalyzerResult:
        """Run analysis on a kernel.

        Args:
            kernel_metadata: KernelMetadata from loader
            config: KernelConfig from launcher

        Returns:
            AnalyzerResult with diagnostics
        """
        pass

    def _safe_analyze(self, kernel_metadata: Any, config: Any) -> AnalyzerResult:
        """Safely run analysis with error handling."""
        result = AnalyzerResult(analyzer_name=self.name, passed=True)

        try:
            inner_result = self.analyze(kernel_metadata, config)
            result.diagnostics = inner_result.diagnostics
            result.errors = inner_result.errors
            result.passed = inner_result.passed
        except Exception as e:
            self.logger.error(f"Analyzer {self.name} failed: {e}", exc_info=True)
            result.passed = False
            result.add_error(str(e))

        return result
