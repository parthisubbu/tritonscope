"""Phase 0 baseline tests: packaging and CLI structure."""

import pytest
from pathlib import Path
from tritonscope import __version__, AnalysisConfig


def test_version():
    """Verify version is accessible."""
    assert __version__ == "0.1.0"
    assert isinstance(__version__, str)


def test_analysis_config():
    """Test AnalysisConfig placeholder class."""
    config = AnalysisConfig(kernel_func=lambda: None, triton_config={"num_warps": 4})
    assert config.kernel_func is not None
    assert config.triton_config == {"num_warps": 4}


def test_cli_imports():
    """Verify CLI module imports correctly."""
    from tritonscope.cli import app, analyze, doctor, version
    assert app is not None
    assert callable(analyze)
    assert callable(doctor)
    assert callable(version)


def test_package_structure():
    """Verify package structure exists."""
    base = Path(__file__).parent.parent
    assert (base / "tritonscope" / "__init__.py").exists()
    assert (base / "tritonscope" / "__version__.py").exists()
    assert (base / "tritonscope" / "cli.py").exists()
    assert (base / "pyproject.toml").exists()
    assert (base / ".github" / "workflows" / "ci.yml").exists()


def test_pyproject_metadata():
    """Verify pyproject.toml has correct metadata."""
    import tomllib
    from pathlib import Path

    base = Path(__file__).parent.parent
    with open(base / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert pyproject["project"]["name"] == "tritonscope"
    assert pyproject["project"]["version"] == "0.1.0"
    assert "triton" in str(pyproject["project"]["optional-dependencies"]["gpu"]).lower()
    assert pyproject["project"]["requires-python"] == ">=3.10"


def test_dependencies_present():
    """Verify core dependencies are listed."""
    import tomllib
    from pathlib import Path

    base = Path(__file__).parent.parent
    with open(base / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject["project"]["dependencies"]
    dep_names = [d.split()[0].lower() for d in deps]

    required = ["pydantic", "typer", "rich", "packaging", "psutil"]
    for req in required:
        assert any(req in d for d in dep_names), f"Missing {req}"
