# Contributing to TrionScope

## Development Setup

```bash
# Clone repo
git clone git@github.com:parthisubbu/tritonscope.git
cd tritonscope

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format & lint
black tritonscope tests
ruff check tritonscope tests
```

## Testing

- Unit tests: `tests/test_phase*.py`
- Run all: `pytest tests/ -v`
- Run specific: `pytest tests/test_phase2.py -v`
- Coverage: `pytest --cov=tritonscope tests/`

## Roadmap

**Completed:** Phases 0-3 (Core infrastructure, loading, analysis)
**In Progress:** Phases 4-9 (Rules, outputs, API, release)

## Code Style

- Black formatting
- Ruff linting
- Type hints (mypy-compatible)
- Docstrings on all public functions

## Pull Requests

1. Fork and branch (`git checkout -b feature/my-feature`)
2. Add tests for any new functionality
3. Run tests and linting locally
4. Push and submit PR with description
5. Wait for CI to pass

---

Questions? Open an issue or contact the maintainers.
