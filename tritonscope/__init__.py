"""TritonScope: GPU kernel analysis and optimization for Triton kernels.

Diagnose Triton GPU kernel performance issues in seconds, in plain English, for free.
"""

from tritonscope.__version__ import __version__

__all__ = ["__version__", "analyze", "AnalysisConfig"]


class AnalysisConfig:
    """Configuration for kernel analysis (placeholder for Phase 2)."""

    def __init__(self, kernel_func=None, triton_config=None):
        self.kernel_func = kernel_func
        self.triton_config = triton_config or {}


def analyze(config: AnalysisConfig):
    """Main analysis entry point. Full implementation in Phase 2+.

    Args:
        config: AnalysisConfig instance with kernel and configuration.

    Returns:
        Analysis report (schema defined in Phase 3).
    """
    raise NotImplementedError("analyze() implementation coming in Phase 2")
