"""GPU and hardware detection for TrionScope."""

import sys
import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about detected GPU."""

    available: bool
    name: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    device_count: int = 0


def detect_cuda() -> dict:
    """Detect CUDA availability and version.

    Returns:
        Dict with keys: available (bool), version (str), driver_version (str)
    """
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "version": torch.version.cuda or "Unknown",
                "driver_version": _get_cuda_driver_version(),
            }
    except ImportError:
        pass

    return {
        "available": False,
        "version": None,
        "driver_version": None,
    }


def detect_gpu() -> GPUInfo:
    """Detect GPU details.

    Returns:
        GPUInfo object with GPU details or available=False if no GPU found.
    """
    info = GPUInfo(available=False)

    try:
        import torch

        if not torch.cuda.is_available():
            return info

        info.available = True
        info.device_count = torch.cuda.device_count()
        info.name = torch.cuda.get_device_name(0)
        info.memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Compute capability
        major, minor = torch.cuda.get_device_capability(0)
        info.compute_capability = f"{major}.{minor}"

        # CUDA version
        cuda_info = detect_cuda()
        info.cuda_version = cuda_info.get("version")
        info.cuda_driver_version = cuda_info.get("driver_version")

    except Exception as e:
        # Silently fail if torch/cuda not available or error occurs
        pass

    return info


def detect_triton() -> dict:
    """Detect Triton installation and version.

    Returns:
        Dict with keys: installed (bool), version (str)
    """
    try:
        import triton

        return {
            "installed": True,
            "version": triton.__version__,
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
        }


def detect_pytorch() -> dict:
    """Detect PyTorch installation and version.

    Returns:
        Dict with keys: installed (bool), version (str)
    """
    try:
        import torch

        return {
            "installed": True,
            "version": torch.__version__,
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
        }


def detect_rocm() -> dict:
    """Detect ROCm (AMD GPU support).

    Returns:
        Dict with keys: available (bool), version (str)
    """
    try:
        import torch

        if hasattr(torch, "version") and hasattr(torch.version, "hip"):
            return {
                "available": True,
                "version": torch.version.hip or "Unknown",
            }
    except Exception:
        pass

    return {
        "available": False,
        "version": None,
    }


def _get_cuda_driver_version() -> Optional[str]:
    """Get CUDA driver version from nvidia-smi or torch.

    Returns:
        Driver version string or None if not found.
    """
    try:
        import torch

        # Try to get from torch first
        if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
            return torch.version.cuda

        # Fallback: parse nvidia-smi output
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def get_system_info() -> dict:
    """Get comprehensive system information.

    Returns:
        Dict with keys: python, platform, cpu_count, gpu, cuda, pytorch, triton, rocm
    """
    try:
        import psutil

        cpu_count = psutil.cpu_count()
    except ImportError:
        cpu_count = None

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "gpu": detect_gpu(),
        "cuda": detect_cuda(),
        "pytorch": detect_pytorch(),
        "triton": detect_triton(),
        "rocm": detect_rocm(),
    }
