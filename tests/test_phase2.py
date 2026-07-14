"""Phase 2 tests: Kernel loading and launch config generation."""

import pytest
import tempfile
from pathlib import Path

from tritonscope.core.loader import KernelLoader, KernelMetadata, KernelParameter
from tritonscope.core.launcher import LaunchConfigGenerator, KernelConfig


class TestKernelLoader:
    """Test kernel detection and loading."""

    @pytest.fixture
    def softmax_kernel_file(self):
        """Create a temporary kernel file for testing."""
        code = '''
import triton
import triton.language as tl

@triton.jit
def softmax(X, Y, BLOCK_SIZE: tl.constexpr):
    """Softmax kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    y = tl.softmax(x)
    tl.store(Y + offsets, y)

@triton.jit
def relu(X, Y):
    """ReLU activation."""
    idx = tl.program_id(0)
    x = tl.load(X + idx)
    y = tl.maximum(x, 0.0)
    tl.store(Y + idx, y)
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_loader_init(self, softmax_kernel_file):
        """Test loader initialization."""
        loader = KernelLoader(softmax_kernel_file)
        assert loader.file_path == softmax_kernel_file
        assert loader.source is not None
        assert loader.tree is not None

    def test_loader_init_missing_file(self):
        """Test loader with missing file."""
        with pytest.raises(FileNotFoundError):
            KernelLoader(Path("/nonexistent/path.py"))

    def test_find_all_kernels(self, softmax_kernel_file):
        """Test finding all kernels in a file."""
        loader = KernelLoader(softmax_kernel_file)
        kernels = loader.find_all_kernels()
        assert len(kernels) == 2
        assert any(k.name == "softmax" for k in kernels)
        assert any(k.name == "relu" for k in kernels)

    def test_find_specific_kernel(self, softmax_kernel_file):
        """Test finding a specific kernel by name."""
        loader = KernelLoader(softmax_kernel_file)
        kernel = loader.find_kernel("softmax")
        assert kernel is not None
        assert kernel.name == "softmax"
        assert kernel.file_path == softmax_kernel_file

    def test_kernel_metadata_structure(self, softmax_kernel_file):
        """Test KernelMetadata structure."""
        loader = KernelLoader(softmax_kernel_file)
        kernel = loader.find_kernel("softmax")
        assert isinstance(kernel, KernelMetadata)
        assert kernel.name == "softmax"
        assert kernel.line_number > 0
        assert kernel.parameters is not None
        assert kernel.docstring is not None

    def test_kernel_parameters_extraction(self, softmax_kernel_file):
        """Test parameter extraction from kernel."""
        loader = KernelLoader(softmax_kernel_file)
        kernel = loader.find_kernel("softmax")

        assert len(kernel.parameters) == 3
        param_names = [p.name for p in kernel.parameters]
        assert "X" in param_names
        assert "Y" in param_names
        assert "BLOCK_SIZE" in param_names

    def test_constexpr_detection(self, softmax_kernel_file):
        """Test constexpr parameter detection."""
        loader = KernelLoader(softmax_kernel_file)
        kernel = loader.find_kernel("softmax")

        constexprs = kernel.constexprs
        assert len(constexprs) == 1
        assert constexprs[0].name == "BLOCK_SIZE"

    def test_tensor_params_extraction(self, softmax_kernel_file):
        """Test tensor parameter extraction."""
        loader = KernelLoader(softmax_kernel_file)
        kernel = loader.find_kernel("softmax")

        tensor_params = kernel.tensor_params
        assert len(tensor_params) == 2
        assert all(not p.is_constexpr for p in tensor_params)


class TestLaunchConfigGenerator:
    """Test launch config generation."""

    @pytest.fixture
    def softmax_metadata(self, softmax_kernel_file):
        """Create kernel metadata for testing."""
        loader = KernelLoader(softmax_kernel_file)
        return loader.find_kernel("softmax")

    @pytest.fixture
    def softmax_kernel_file(self):
        """Create a temporary kernel file."""
        code = '''
import triton
import triton.language as tl

@triton.jit
def softmax(X, Y, BLOCK_SIZE: tl.constexpr):
    """Softmax kernel."""
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_parse_user_config(self):
        """Test parsing user config string."""
        config_str = "BLOCK_SIZE=512 num_warps=8"
        config = LaunchConfigGenerator.parse_user_config(config_str)

        assert config["BLOCK_SIZE"] == 512
        assert config["num_warps"] == 8
        assert isinstance(config["BLOCK_SIZE"], int)

    def test_parse_user_config_mixed_types(self):
        """Test parsing config with mixed types."""
        config_str = "BLOCK_SIZE=256 learning_rate=0.001 name=softmax"
        config = LaunchConfigGenerator.parse_user_config(config_str)

        assert config["BLOCK_SIZE"] == 256
        assert config["learning_rate"] == 0.001
        assert config["name"] == "softmax"

    def test_parse_user_config_empty(self):
        """Test parsing empty config."""
        config = LaunchConfigGenerator.parse_user_config(None)
        assert config == {}

    def test_parse_user_config_invalid_format(self):
        """Test parsing invalid config format."""
        with pytest.raises(ValueError):
            LaunchConfigGenerator.parse_user_config("INVALID_FORMAT")

    def test_generate_config_basic(self, softmax_metadata):
        """Test basic config generation."""
        config = LaunchConfigGenerator.generate_config(
            softmax_metadata,
            {"BLOCK_SIZE": 512}
        )

        assert isinstance(config, KernelConfig)
        assert config.kernel_name == "softmax"
        assert config.constexpr_values["BLOCK_SIZE"] == 512
        assert config.num_warps == 4
        assert config.num_stages == 3

    def test_generate_config_missing_constexpr(self, softmax_metadata):
        """Test config generation with missing constexpr."""
        with pytest.raises(ValueError):
            LaunchConfigGenerator.generate_config(softmax_metadata, {})

    def test_kernel_config_to_dict(self, softmax_metadata):
        """Test KernelConfig.to_dict()."""
        config = LaunchConfigGenerator.generate_config(
            softmax_metadata,
            {"BLOCK_SIZE": 256}
        )

        config_dict = config.to_dict()
        assert config_dict["BLOCK_SIZE"] == 256
        assert config_dict["num_warps"] == 4
        assert config_dict["num_stages"] == 3


class TestLaunchConfigIntegration:
    """Integration tests for loader + launcher."""

    def test_end_to_end_kernel_loading_and_config(self):
        """Test complete workflow: load kernel -> generate config."""
        code = '''
import triton
import triton.language as tl

@triton.jit
def matmul(A, B, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Matrix multiply kernel."""
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            kernel_file = Path(f.name)

        try:
            # Load kernel
            loader = KernelLoader(kernel_file)
            kernel = loader.find_kernel("matmul")
            assert kernel is not None

            # Generate config
            config = LaunchConfigGenerator.generate_config(
                kernel,
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
            )

            assert config.kernel_name == "matmul"
            assert len(config.constexpr_values) == 3
        finally:
            kernel_file.unlink()
