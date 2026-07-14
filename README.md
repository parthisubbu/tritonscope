# TrionScope

**Diagnose Triton GPU kernel performance issues in seconds, in plain English, for free.**

Instead of guessing at configs, waiting 10+ minutes for an autotuner, or paying for NVIDIA Nsight Compute, TrionScope reads the artifacts Triton already produces (PTX assembly, compilation metadata, compiled-kernel handles) and fires rule-based diagnostics that tell you *what's wrong* and *what to change*.

## Status

🚧 **Phase 0 (v0.1.0):** Package structure and CI pipeline. Core analysis in Phase 2+.

## Install

```bash
pip install -e .              # Install from source
pip install -e ".[gpu]"       # Also install torch + triton for live collection
pip install -e ".[dev]"       # Install dev tools (pytest, black, ruff)
```

## Quick Start (Phase 2+)

```bash
tritonscope analyze my_softmax.py --kernel softmax --config "BLOCK_SIZE=512"
tritonscope doctor  # System diagnostics
```

## How It Works (Full Implementation Phase 2+)

```
kernel.py → [Collector] runs kernel, captures PTX / metadata
          → [Parsers] extract metrics (register counts, memory, etc.)
          → [Occupancy Calculator] compute theoretical occupancy
          → [Rule Engine] fire diagnostics (30+ rules, categorized)
          → [Reporter] plain-English diagnosis + fixes
```

## Features (Roadmap)

- **Phase 1 (Weeks 2-3):** PyPI publishing, GPU auto-detection, `tritonscope doctor`
- **Phase 2 (Weeks 4-5):** Kernel loading, launch config generation, graceful errors
- **Phase 3 (Weeks 6-8):** Analyzer isolation, structured diagnostics, no crashes
- **Phase 4 (Weeks 9-11):** Universal kernel support, 30+ rules, rule registry
- **Phase 5 (Weeks 12-13):** JSON, HTML, Markdown, terminal output formats
- **Phase 6 (Week 14):** Performance optimization, compilation caching, logging
- **Phase 7 (Weeks 15-16):** Public API, 80+ tests, GitHub Actions CI
- **Phase 8 (Weeks 17-18):** Documentation, examples, CHANGELOG
- **Phase 9 (Week 19):** v1.0.0 release, PyPI announcement

## Development

```bash
# Run tests
pytest tests/ -v

# Lint & format
black tritonscope tests
ruff check tritonscope tests

# Build distribution
python -m build
```

## Architecture

```
tritonscope/
  ├── __init__.py          # Public API
  ├── __version__.py       # Version info
  ├── cli.py              # CLI (typer)
  ├── core/
  │   ├── loader.py       # Kernel detection & parsing
  │   └── launcher.py     # Auto launch config generation
  ├── analyzers/
  │   ├── ast_analyzer.py     # Python AST analysis
  │   ├── ptx_analyzer.py     # PTX assembly parsing
  │   ├── ttir_analyzer.py    # Triton IR analysis (optional)
  │   └── runtime_analyzer.py # Occupancy & math
  ├── rules/
  │   ├── registry.py     # Rule registry (30+ rules)
  │   └── (individual rules)
  └── output/
      ├── json_formatter.py
      ├── terminal_formatter.py
      ├── html_formatter.py
      └── markdown_formatter.py

tests/
  ├── test_phase0.py      # Package structure tests
  ├── test_loader.py      # Kernel loading tests
  ├── test_analyzers.py   # Analyzer isolation tests
  └── test_pipeline.py    # End-to-end tests
```

## Rules (Phase 4+)

Over 30 rules categorized by issue type:

| Category | Rules |
|----------|-------|
| **Occupancy** | LOW_OCCUPANCY, CANNOT_LAUNCH |
| **Memory** | REGISTER_SPILLING, SHARED_MEMORY_PRESSURE, BANDWIDTH_UNDERUTILIZED |
| **Registers** | HIGH_REGISTER_USAGE |
| **Config** | NUM_STAGES_HIGH, NUM_STAGES_LOW, NUM_WARPS_LOW, BLOCK_SIZE_UNALIGNED |
| **Compiler** | UNOPTIMIZED_CODE_PATTERN |

Each rule includes:
- **basis:** artifact (from compiler), computed (from math), or heuristic
- **severity:** info, warning, or error
- **suggestion:** how to fix it

## Known Limitations

- Live collection requires NVIDIA GPU; analyzing saved bundles works anywhere
- Occupancy is *theoretical* (resource-fit); real performance also depends on memory patterns and instruction mix
- Heuristic rules are rules of thumb; evaluation study planned for Phase 10+

## About

Built as a portfolio project for GPU kernel optimization. Inspired by NVIDIA Nsight Compute but aiming for speed, simplicity, and free access.

**Author:** Parthi Subbu  
**License:** MIT  
**Repo:** [github.com/parthisubbu/tritonscope](https://github.com/parthisubbu/tritonscope)
