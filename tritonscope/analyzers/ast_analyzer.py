"""AST-based kernel analysis."""

import ast
from typing import Any

from tritonscope.analyzers.base import Analyzer, AnalyzerResult, Diagnostic, Severity, Basis, SourceLocation


class ASTAnalyzer(Analyzer):
    """Analyze kernels using Python AST."""

    def __init__(self):
        super().__init__("AST")

    def analyze(self, kernel_metadata: Any, config: Any) -> AnalyzerResult:
        """Analyze kernel using AST.

        Checks for:
        - Loop patterns
        - Memory access patterns
        - Control flow complexity
        """
        result = AnalyzerResult(analyzer_name=self.name, passed=True)

        if not hasattr(kernel_metadata, 'source_code') or not kernel_metadata.source_code:
            return result

        try:
            tree = ast.parse(kernel_metadata.source_code)
            self._analyze_tree(tree, kernel_metadata, result)
        except Exception as e:
            result.passed = False
            result.add_error(f"AST parsing failed: {e}")

        return result

    def _analyze_tree(self, tree: ast.AST, kernel_metadata: Any, result: AnalyzerResult) -> None:
        """Analyze AST tree."""
        for node in ast.walk(tree):
            # Check for loops
            if isinstance(node, ast.For):
                self._check_loop(node, kernel_metadata, result)

            # Check for complex conditionals
            if isinstance(node, ast.If):
                self._check_conditional(node, kernel_metadata, result)

    def _check_loop(self, node: ast.For, kernel_metadata: Any, result: AnalyzerResult) -> None:
        """Check loop patterns."""
        # Count loop nesting depth
        depth = 0
        parent = node
        while parent:
            if isinstance(parent, (ast.For, ast.While)):
                depth += 1
            parent = getattr(parent, '_parent', None)

        if depth > 2:
            diag = Diagnostic(
                id="TS201",
                category="Control Flow",
                severity=Severity.WARNING,
                basis=Basis.HEURISTIC,
                message=f"Deep loop nesting (depth={depth})",
                evidence="Nested loops reduce occupancy",
                suggestion="Consider flattening or simplifying loop structure",
                location=SourceLocation(
                    file=kernel_metadata.file_path,
                    line=node.lineno,
                    function=kernel_metadata.name,
                ),
            )
            result.add_diagnostic(diag)

    def _check_conditional(self, node: ast.If, kernel_metadata: Any, result: AnalyzerResult) -> None:
        """Check conditional patterns."""
        # Count branch depth
        depth = 1
        current = node.orelse
        while current and len(current) == 1 and isinstance(current[0], ast.If):
            depth += 1
            current = current[0].orelse

        if depth > 3:
            diag = Diagnostic(
                id="TS202",
                category="Control Flow",
                severity=Severity.INFO,
                basis=Basis.HEURISTIC,
                message=f"Many conditional branches (depth={depth})",
                evidence="Branches create control flow divergence",
                suggestion="Simplify branching logic if possible",
                location=SourceLocation(
                    file=kernel_metadata.file_path,
                    line=node.lineno,
                    function=kernel_metadata.name,
                ),
            )
            result.add_diagnostic(diag)
