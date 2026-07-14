"""Version and compatibility checking for TrionScope."""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
from packaging import version


@dataclass
class CompatibilityCheck:
    """Result of a single compatibility check."""

    name: str
    passed: bool
    actual: Optional[str] = None
    required: Optional[str] = None
    message: Optional[str] = None

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        msg = self.message or ""
        if self.actual and self.required and not self.passed:
            msg = f"Required: {self.required}, Got: {self.actual}"
        return f"{status} {self.name:<30} {msg}"


@dataclass
class CompatibilityReport:
    """Full compatibility report."""

    passed: bool
    checks: List[CompatibilityCheck]
    failures: List[str]
    warnings: List[str]

    def summary(self) -> str:
        """Human-readable summary."""
        if self.passed:
            return "✓ All compatibility checks passed"
        return f"✗ {len(self.failures)} compatibility issue(s) found"


def check_python_version() -> CompatibilityCheck:
    """Check Python version (3.10+ required)."""
    min_version = "3.10"
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    current_tuple = (sys.version_info.major, sys.version_info.minor)
    required_tuple = (3, 10)

    passed = current_tuple >= required_tuple
    return CompatibilityCheck(
        name="Python",
        passed=passed,
        actual=current,
        required=f">= {min_version}",
    )


def check_pytorch_version() -> CompatibilityCheck:
    """Check PyTorch version (2.0+ required if installed)."""
    try:
        import torch

        min_version = "2.0"
        current = torch.__version__.split("+")[0]  # Remove +cu118 suffix
        passed = version.parse(current) >= version.parse(min_version)
        return CompatibilityCheck(
            name="PyTorch",
            passed=passed,
            actual=current,
            required=f">= {min_version}",
        )
    except ImportError:
        return CompatibilityCheck(
            name="PyTorch",
            passed=True,
            actual="Not installed",
            message="(Optional - install with: pip install tritonscope[gpu])",
        )


def check_triton_version() -> CompatibilityCheck:
    """Check Triton version (must be 3.6.0)."""
    try:
        import triton

        required_version = "3.6.0"
        current = triton.__version__
        passed = version.parse(current) == version.parse(required_version)
        return CompatibilityCheck(
            name="Triton",
            passed=passed,
            actual=current,
            required=required_version,
        )
    except ImportError:
        return CompatibilityCheck(
            name="Triton",
            passed=True,
            actual="Not installed",
            message="(Optional - install with: pip install tritonscope[gpu])",
        )


def check_cuda_availability() -> CompatibilityCheck:
    """Check CUDA availability (if torch is installed)."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return CompatibilityCheck(
                name="CUDA",
                passed=True,
                actual=cuda_version,
                message=f"GPU: {torch.cuda.get_device_name(0)}",
            )
        else:
            return CompatibilityCheck(
                name="CUDA",
                passed=True,
                actual="Not available",
                message="(CPU-only mode - GPU analysis requires CUDA)",
            )
    except ImportError:
        return CompatibilityCheck(
            name="CUDA",
            passed=True,
            actual="Not checked (PyTorch not installed)",
            message="(Optional)",
        )


def check_compiler_availability() -> CompatibilityCheck:
    """Check C/C++ compiler availability (for Triton compilation)."""
    import shutil

    # Check for common compilers
    compilers = {
        "gcc": "GCC",
        "clang": "Clang",
        "cl": "MSVC",  # Windows
    }

    found_compiler = None
    for cmd, name in compilers.items():
        if shutil.which(cmd):
            found_compiler = name
            break

    if found_compiler:
        return CompatibilityCheck(
            name="C++ Compiler",
            passed=True,
            actual=found_compiler,
            message="Available",
        )
    else:
        return CompatibilityCheck(
            name="C++ Compiler",
            passed=False,
            actual="Not found",
            message="Required for Triton kernel compilation",
        )


def check_cache_permissions() -> CompatibilityCheck:
    """Check Triton cache directory permissions."""
    from pathlib import Path

    cache_dir = Path.home() / ".triton" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try to write a test file
        test_file = cache_dir / ".tritonscope_test"
        test_file.write_text("test")
        test_file.unlink()

        return CompatibilityCheck(
            name="Cache Directory",
            passed=True,
            actual=str(cache_dir),
            message="Writable",
        )
    except Exception as e:
        return CompatibilityCheck(
            name="Cache Directory",
            passed=False,
            actual=str(cache_dir),
            message=f"Not writable: {e}",
        )


def run_compatibility_check() -> CompatibilityReport:
    """Run all compatibility checks.

    Returns:
        CompatibilityReport with all checks and summary.
    """
    checks = [
        check_python_version(),
        check_pytorch_version(),
        check_triton_version(),
        check_cuda_availability(),
        check_compiler_availability(),
        check_cache_permissions(),
    ]

    failures = [
        check.name for check in checks if not check.passed and "Optional" not in (check.message or "")
    ]

    warnings = [
        check.name for check in checks if not check.passed and "Optional" in (check.message or "")
    ]

    return CompatibilityReport(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )
