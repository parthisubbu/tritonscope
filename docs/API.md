# TrionScope Public API

Complete programmatic interface for kernel analysis.

## Quick Start

```python
from tritonscope import analyze, AnalysisConfig

# Create analysis config
config = AnalysisConfig(
    kernel_file="softmax.py",
    kernel_name="softmax_kernel",
    kernel_config="BLOCK_SIZE=256 num_warps=4"
)

# Run analysis
report = analyze(config)

# Access diagnostics
for diag in report.all_diagnostics:
    print(f"{diag.id}: {diag.message}")
```

## API Reference

### High-Level API

#### `analyze(config: AnalysisConfig) -> AnalysisReport`

Run full kernel analysis.

**Args:**
- `config` (AnalysisConfig): Analysis configuration

**Returns:**
- `AnalysisReport`: Complete analysis results

**Example:**
```python
report = analyze(config)
print(report.summary())
```

### Core Modules

#### `tritonscope.core.loader`

**`KernelLoader(file_path)`**
- Load and detect Triton kernels from Python files

**Methods:**
- `find_all_kernels()` → List[KernelMetadata]
- `find_kernel(name: str)` → Optional[KernelMetadata]
- `load_kernel_function(file_path, kernel_name)` → Optional[Callable]

**Example:**
```python
from tritonscope.core.loader import KernelLoader

loader = KernelLoader("softmax.py")
kernels = loader.find_all_kernels()
softmax = loader.find_kernel("softmax_kernel")

print(softmax.constexprs)  # [BLOCK_SIZE]
print(softmax.tensor_params)  # [X, Y]
```

#### `tritonscope.core.launcher`

**`LaunchConfigGenerator`**
- Generate launch configurations for kernels

**Methods:**
- `generate_config(kernel_metadata, user_config)` → KernelConfig
- `parse_user_config(config_str)` → Dict[str, Any]
- `generate_test_inputs(kernel_metadata, config)` → Dict[str, Tensor]

**Example:**
```python
from tritonscope.core.launcher import LaunchConfigGenerator

config_dict = LaunchConfigGenerator.parse_user_config("BLOCK_SIZE=512 num_warps=8")
config = LaunchConfigGenerator.generate_config(kernel, config_dict)
```

### Analyzers

#### `tritonscope.analyzers`

**Base Class: `Analyzer`**
```python
from tritonscope.analyzers.base import Analyzer, Diagnostic

class Analyzer(ABC):
    def analyze(self, kernel_metadata, config) -> AnalyzerResult:
        pass
```

**Built-in Analyzers:**
- `ASTAnalyzer` — Python AST analysis
- `RuntimeAnalyzer` — Hardware-based occupancy and register analysis

**Example:**
```python
from tritonscope.analyzers.ast_analyzer import ASTAnalyzer

analyzer = ASTAnalyzer()
result = analyzer.analyze(kernel, config)

for diag in result.diagnostics:
    print(f"{diag.id}: {diag.message}")
```

### Diagnostics

#### `tritonscope.analyzers.base.Diagnostic`

```python
@dataclass
class Diagnostic:
    id: str              # TS001, TS002, etc.
    category: str        # Occupancy, Memory, Registers, etc.
    severity: Severity   # info, warning, error
    basis: Basis         # artifact, computed, heuristic
    message: str         # Human-readable message
    evidence: str        # Why this was triggered
    suggestion: str      # How to fix it
    location: Optional[SourceLocation]
```

**Severity Levels:**
- `Severity.INFO` — Informational (low priority)
- `Severity.WARNING` — Warning (medium priority)
- `Severity.ERROR` — Error (high priority)

**Basis:**
- `Basis.ARTIFACT` — Observed in compiled output
- `Basis.COMPUTED` — From deterministic math
- `Basis.HEURISTIC` — Rule of thumb

### Pipeline

#### `tritonscope.pipeline`

**`AnalysisPipeline`**
- Orchestrate all analyzers

**Methods:**
- `analyze(kernel_metadata, config)` → AnalysisReport

**Example:**
```python
from tritonscope.pipeline import AnalysisPipeline

pipeline = AnalysisPipeline()
report = pipeline.analyze(kernel, config)

print(f"Errors: {report.error_count}")
print(f"Warnings: {report.warning_count}")
```

### Output Formatters

#### `tritonscope.output.json_formatter`

**`JSONFormatter`**
- Format reports as JSON

**Methods:**
- `format(report) -> str` — Get JSON string
- `to_dict(report) -> Dict` — Get dictionary
- `write_file(report, path)` — Write to file

**Example:**
```python
from tritonscope.output.json_formatter import JSONFormatter

json_str = JSONFormatter.format(report)
JSONFormatter.write_file(report, Path("report.json"))
```

#### `tritonscope.output.terminal_formatter`

**`TerminalFormatter`**
- Format reports for terminal with colors

**Example:**
```python
from tritonscope.output.terminal_formatter import TerminalFormatter
from rich.console import Console

formatter = TerminalFormatter(Console())
formatter.format(report)
```

#### `tritonscope.output.markdown_formatter`

**`MarkdownFormatter`**
- Format reports as Markdown

**Example:**
```python
from tritonscope.output.markdown_formatter import MarkdownFormatter

md_str = MarkdownFormatter.format(report)
print(md_str)
```

### Hardware Detection

#### `tritonscope.hardware`

**Functions:**
- `detect_gpu()` → GPUInfo
- `detect_cuda()` → Dict
- `detect_pytorch()` → Dict
- `detect_triton()` → Dict
- `get_system_info()` → Dict

**Example:**
```python
from tritonscope.hardware import detect_gpu

gpu = detect_gpu()
print(f"GPU: {gpu.name}, Compute Capability: {gpu.compute_capability}")
```

### Compatibility Checking

#### `tritonscope.compatibility`

**Functions:**
- `run_compatibility_check()` → CompatibilityReport

**Example:**
```python
from tritonscope.compatibility import run_compatibility_check

report = run_compatibility_check()
print(report.summary())
```

## Complete Example

```python
from pathlib import Path
from tritonscope.core.loader import KernelLoader
from tritonscope.core.launcher import LaunchConfigGenerator
from tritonscope.pipeline import AnalysisPipeline
from tritonscope.output.json_formatter import JSONFormatter
from tritonscope.output.terminal_formatter import TerminalFormatter

# 1. Load kernel
loader = KernelLoader(Path("softmax.py"))
kernel = loader.find_kernel("softmax_kernel")

# 2. Parse user config
user_config = LaunchConfigGenerator.parse_user_config("BLOCK_SIZE=512 num_warps=8")

# 3. Generate launch config
config = LaunchConfigGenerator.generate_config(kernel, user_config)

# 4. Run analysis
pipeline = AnalysisPipeline()
report = pipeline.analyze(kernel, config)

# 5. Output results
print(report.summary())

# JSON output
json_str = JSONFormatter.format(report)
JSONFormatter.write_file(report, Path("report.json"))

# Terminal output
TerminalFormatter().format(report)
```

## Error Handling

All analyzers are isolated and wrapped in error handlers:

```python
from tritonscope.pipeline import AnalysisPipeline

pipeline = AnalysisPipeline()
report = pipeline.analyze(kernel, config)

# Check if any analyzer failed
for result in report.analyzer_results:
    if not result.passed:
        print(f"Analyzer {result.analyzer_name} failed:")
        for error in result.errors:
            print(f"  - {error}")
```

## Performance

- Kernel loading: ~10ms
- Config generation: ~5ms
- AST analysis: ~20ms
- Runtime analysis: ~10ms
- Total typical: <100ms

## Logging

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tritonscope")
```

---

**TrionScope v0.1.0** — Diagnose Triton GPU kernel performance in seconds.
