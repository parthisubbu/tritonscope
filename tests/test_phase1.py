"""Phase 1 tests: GPU detection and compatibility checking."""

import sys
import pytest
from tritonscope.hardware import (
    detect_cuda,
    detect_gpu,
    detect_pytorch,
    detect_triton,
    detect_rocm,
    GPUInfo,
)
from tritonscope.compatibility import (
    check_python_version,
    check_pytorch_version,
    check_triton_version,
    check_cuda_availability,
    check_cache_permissions,
    run_compatibility_check,
    CompatibilityCheck,
)


class TestHardwareDetection:
    """Test GPU and hardware detection."""

    def test_detect_cuda_returns_dict(self):
        """CUDA detection should return a dict."""
        result = detect_cuda()
        assert isinstance(result, dict)
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_detect_gpu_returns_gpuinfo(self):
        """GPU detection should return GPUInfo."""
        result = detect_gpu()
        assert isinstance(result, GPUInfo)
        assert isinstance(result.available, bool)

    def test_detect_pytorch_returns_dict(self):
        """PyTorch detection should return a dict."""
        result = detect_pytorch()
        assert isinstance(result, dict)
        assert "installed" in result
        assert isinstance(result["installed"], bool)

    def test_detect_triton_returns_dict(self):
        """Triton detection should return a dict."""
        result = detect_triton()
        assert isinstance(result, dict)
        assert "installed" in result
        assert isinstance(result["installed"], bool)

    def test_detect_rocm_returns_dict(self):
        """ROCm detection should return a dict."""
        result = detect_rocm()
        assert isinstance(result, dict)
        assert "available" in result
        assert isinstance(result["available"], bool)


class TestCompatibilityChecks:
    """Test version compatibility checking."""

    def test_python_version_check(self):
        """Python version check should validate 3.10+."""
        check = check_python_version()
        assert isinstance(check, CompatibilityCheck)
        assert check.name == "Python"
        # Current Python should pass (test runs on Python 3.10+)
        assert check.passed is True
        assert check.actual is not None

    def test_pytorch_version_check_when_not_installed(self):
        """PyTorch check should not fail if not installed."""
        check = check_pytorch_version()
        assert isinstance(check, CompatibilityCheck)
        assert check.name == "PyTorch"
        # Should not fail even if not installed
        assert check.passed is True

    def test_triton_version_check_when_not_installed(self):
        """Triton check should not fail if not installed."""
        check = check_triton_version()
        assert isinstance(check, CompatibilityCheck)
        assert check.name == "Triton"
        # Should not fail even if not installed
        assert check.passed is True

    def test_cuda_availability_check(self):
        """CUDA check should return a check object."""
        check = check_cuda_availability()
        assert isinstance(check, CompatibilityCheck)
        assert check.name == "CUDA"
        assert check.passed is True  # Should not fail even if not available

    def test_cache_permissions_check(self):
        """Cache permissions check should verify write access."""
        check = check_cache_permissions()
        assert isinstance(check, CompatibilityCheck)
        assert check.name == "Cache Directory"
        # Should pass if we can write to cache
        assert check.passed is True

    def test_compatibility_report_structure(self):
        """Full compatibility report should have correct structure."""
        report = run_compatibility_check()
        assert report.passed is not None
        assert isinstance(report.checks, list)
        assert len(report.checks) > 0
        assert isinstance(report.failures, list)
        assert isinstance(report.warnings, list)

    def test_compatibility_report_all_checks_present(self):
        """Report should include all required checks."""
        report = run_compatibility_check()
        check_names = {check.name for check in report.checks}

        required_checks = {
            "Python",
            "PyTorch",
            "Triton",
            "CUDA",
            "C++ Compiler",
            "Cache Directory",
        }
        assert required_checks.issubset(check_names)

    def test_compatibility_check_string_representation(self):
        """CompatibilityCheck should have readable string output."""
        check = check_python_version()
        output = str(check)
        assert check.name in output
        assert ("✓" in output or "✗" in output)


class TestSystemInfo:
    """Test system information gathering."""

    def test_get_system_info(self):
        """get_system_info should return comprehensive dict."""
        from tritonscope.hardware import get_system_info

        info = get_system_info()
        assert isinstance(info, dict)
        assert "python" in info
        assert "platform" in info
        assert "gpu" in info
        assert "cuda" in info
        assert "pytorch" in info
        assert "triton" in info
        assert "rocm" in info

    def test_system_info_gpu_data(self):
        """GPU data in system info should be GPUInfo."""
        from tritonscope.hardware import get_system_info

        info = get_system_info()
        assert isinstance(info["gpu"], GPUInfo)

    def test_system_info_cuda_data(self):
        """CUDA data should be a dict."""
        from tritonscope.hardware import get_system_info

        info = get_system_info()
        assert isinstance(info["cuda"], dict)
        assert "available" in info["cuda"]
