"""Kernel loading and detection for TrionScope."""

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional


@dataclass
class KernelParameter:
    """Metadata for a kernel parameter."""

    name: str
    annotation: Optional[str] = None
    is_constexpr: bool = False
    default_value: Optional[Any] = None


@dataclass
class KernelMetadata:
    """Metadata extracted from a @triton.jit kernel."""

    name: str
    file_path: Path
    line_number: int
    parameters: List[KernelParameter]
    docstring: Optional[str] = None
    source_code: Optional[str] = None

    @property
    def constexprs(self) -> List[KernelParameter]:
        """Get only constexpr parameters."""
        return [p for p in self.parameters if p.is_constexpr]

    @property
    def tensor_params(self) -> List[KernelParameter]:
        """Get only tensor parameters (non-constexpr)."""
        return [p for p in self.parameters if not p.is_constexpr]


class KernelLoader:
    """Load and analyze Triton kernels from Python files."""

    def __init__(self, file_path: Path):
        """Initialize loader for a specific file.

        Args:
            file_path: Path to Python file containing kernels
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {file_path}")

        self.source = self.file_path.read_text()
        self.tree = ast.parse(self.source)

    def find_all_kernels(self) -> List[KernelMetadata]:
        """Find all @triton.jit decorated functions.

        Returns:
            List of KernelMetadata for each kernel found.
        """
        kernels = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_triton_jit(node):
                    metadata = self._extract_metadata(node)
                    if metadata:
                        kernels.append(metadata)
        return kernels

    def find_kernel(self, kernel_name: str) -> Optional[KernelMetadata]:
        """Find a specific kernel by name.

        Args:
            kernel_name: Name of the kernel function.

        Returns:
            KernelMetadata or None if not found.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
                if self._is_triton_jit(node):
                    return self._extract_metadata(node)
        return None

    def _is_triton_jit(self, node: ast.FunctionDef) -> bool:
        """Check if function has @triton.jit decorator."""
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator)
            if 'triton.jit' in decorator_str or decorator_str == 'jit':
                return True
        return False

    def _extract_metadata(self, node: ast.FunctionDef) -> Optional[KernelMetadata]:
        """Extract metadata from a kernel function."""
        try:
            parameters = self._extract_parameters(node)
            source_lines = self.source.split('\n')
            end_line = min(node.end_lineno or node.lineno + 50, len(source_lines))
            source_code = '\n'.join(source_lines[node.lineno - 1:end_line])

            return KernelMetadata(
                name=node.name,
                file_path=self.file_path,
                line_number=node.lineno,
                parameters=parameters,
                docstring=ast.get_docstring(node),
                source_code=source_code,
            )
        except Exception:
            return None

    def _extract_parameters(self, node: ast.FunctionDef) -> List[KernelParameter]:
        """Extract parameter info from function signature."""
        params = []
        for arg in node.args.args:
            if arg.arg == 'self':
                continue

            is_constexpr = self._is_constexpr_param(node, arg.arg)
            param = KernelParameter(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                is_constexpr=is_constexpr,
            )
            params.append(param)

        return params

    def _is_constexpr_param(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if a parameter is declared as constexpr."""
        for arg in func_node.args.args:
            if arg.arg == param_name and arg.annotation:
                ann_str = ast.unparse(arg.annotation).lower()
                if 'constexpr' in ann_str:
                    return True
        return False

    @staticmethod
    def load_kernel_function(file_path: Path, kernel_name: str) -> Optional[Callable]:
        """Dynamically load a kernel function from file."""
        try:
            spec = importlib.util.spec_from_file_location("kernel_module", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, kernel_name, None)
        except Exception:
            pass
        return None
