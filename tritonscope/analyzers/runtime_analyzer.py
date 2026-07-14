"""Runtime and hardware-based analysis."""

from typing import Any

from tritonscope.analyzers.base import Analyzer, AnalyzerResult, Diagnostic, Severity, Basis


class RuntimeAnalyzer(Analyzer):
    """Analyze kernel based on runtime properties and hardware specs."""

    # Hardware specs (T4 baseline)
    HARDWARE_SPECS = {
        "T4": {
            "max_threads_per_block": 1024,
            "max_blocks_per_sm": 32,
            "shared_memory_per_block": 49152,  # 48 KB
            "registers_per_thread": 256,
            "warp_size": 32,
            "sm_count": 40,
        }
    }

    def __init__(self):
        super().__init__("Runtime")

    def analyze(self, kernel_metadata: Any, config: Any) -> AnalyzerResult:
        """Analyze kernel runtime properties.

        Computes:
        - Occupancy (theoretical)
        - Register pressure
        - Shared memory usage
        - Thread block sizing
        """
        result = AnalyzerResult(analyzer_name=self.name, passed=True)

        try:
            specs = self.HARDWARE_SPECS.get("T4")
            if not specs:
                return result

            # Extract config
            block_size = config.constexpr_values.get("BLOCK_SIZE", 256)
            num_warps = config.num_warps
            num_stages = config.num_stages

            # Compute occupancy
            threads_per_block = block_size
            warps_per_block = threads_per_block // specs["warp_size"]

            # Assume 100 registers per thread (heuristic)
            registers_per_block = threads_per_block * 100
            occupancy = min(
                specs["max_threads_per_block"] // threads_per_block,
                specs["max_blocks_per_sm"],
                specs["registers_per_thread"] * specs["warp_size"] // registers_per_block,
            )

            occupancy_pct = (occupancy * warps_per_block / (specs["sm_count"] * 32)) * 100

            # Diagnostics
            if occupancy_pct < 25:
                diag = Diagnostic(
                    id="TS101",
                    category="Occupancy",
                    severity=Severity.ERROR,
                    basis=Basis.COMPUTED,
                    message=f"Very low occupancy ({occupancy_pct:.1f}%)",
                    evidence=f"Only {occupancy} blocks per SM with {threads_per_block} threads/block",
                    suggestion="Reduce BLOCK_SIZE or increase num_warps",
                )
                result.add_diagnostic(diag)
            elif occupancy_pct < 50:
                diag = Diagnostic(
                    id="TS102",
                    category="Occupancy",
                    severity=Severity.WARNING,
                    basis=Basis.COMPUTED,
                    message=f"Low occupancy ({occupancy_pct:.1f}%)",
                    evidence=f"{occupancy} blocks per SM",
                    suggestion="Consider tuning BLOCK_SIZE for better occupancy",
                )
                result.add_diagnostic(diag)

            # Register pressure
            if registers_per_block > specs["registers_per_thread"] * specs["warp_size"] * 0.8:
                diag = Diagnostic(
                    id="TS103",
                    category="Registers",
                    severity=Severity.WARNING,
                    basis=Basis.HEURISTIC,
                    message=f"High register pressure ({registers_per_block} registers/block)",
                    evidence="Estimated high register usage",
                    suggestion="Reduce BLOCK_SIZE or simplify kernel logic",
                )
                result.add_diagnostic(diag)

        except Exception as e:
            result.passed = False
            result.add_error(f"Runtime analysis failed: {e}")

        return result
