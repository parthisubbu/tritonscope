"""Auto-launch configuration generation for Triton kernels."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tritonscope.core.loader import KernelMetadata, KernelParameter


@dataclass
class KernelConfig:
    """Configuration for a kernel launch."""

    kernel_name: str
    constexpr_values: Dict[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 3
    grid_size: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for kernel launch."""
        config = {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }
        config.update(self.constexpr_values)
        return config


class LaunchConfigGenerator:
    """Generate launch configurations for Triton kernels."""

    # Default values for common constexpr parameters
    CONSTEXPR_DEFAULTS = {
        "BLOCK_SIZE": 256,
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_SIZE": 32,
        "num_heads": 8,
        "head_dim": 64,
    }

    # Constraints for constexpr values
    CONSTEXPR_CONSTRAINTS = {
        "BLOCK_SIZE": (32, 1024, 32),  # (min, max, step)
        "BLOCK_M": (32, 256, 32),
        "BLOCK_N": (32, 256, 32),
        "BLOCK_K": (16, 128, 16),
        "num_warps": (1, 32, 1),
        "num_stages": (1, 5, 1),
    }

    @classmethod
    def generate_config(
        cls,
        kernel_metadata: KernelMetadata,
        user_config: Optional[Dict[str, Any]] = None,
    ) -> KernelConfig:
        """Generate a launch config for a kernel.

        Args:
            kernel_metadata: Metadata from kernel loader
            user_config: User-provided config overrides

        Returns:
            KernelConfig ready for launch
        """
        constexpr_values = {}
        user_config = user_config or {}

        # Process each constexpr parameter
        for param in kernel_metadata.constexprs:
            if param.name in user_config:
                # Use user-provided value
                constexpr_values[param.name] = user_config[param.name]
            elif param.name in cls.CONSTEXPR_DEFAULTS:
                # Use default
                constexpr_values[param.name] = cls.CONSTEXPR_DEFAULTS[param.name]
            else:
                # Unknown constexpr - user must provide it
                raise ValueError(
                    f"Unknown constexpr parameter: {param.name}. "
                    f"Please provide via --config {param.name}=value"
                )

        return KernelConfig(
            kernel_name=kernel_metadata.name,
            constexpr_values=constexpr_values,
            num_warps=user_config.get("num_warps", 4),
            num_stages=user_config.get("num_stages", 3),
            grid_size=user_config.get("grid_size"),
        )

    @classmethod
    def generate_test_inputs(
        cls,
        kernel_metadata: KernelMetadata,
        config: KernelConfig,
    ) -> Dict[str, Any]:
        """Generate dummy test inputs for kernel parameters.

        Args:
            kernel_metadata: Kernel metadata
            config: Launch configuration

        Returns:
            Dict of parameter_name -> test_tensor
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for test input generation. "
                "Install with: pip install tritonscope[gpu]"
            )

        inputs = {}

        # Generate test tensors for non-constexpr parameters
        for param in kernel_metadata.tensor_params:
            # Default: create a 1D tensor with 512 elements
            shape = (512,)

            # Try to infer shape from parameter name
            if "weight" in param.name.lower() or "w" == param.name.lower():
                shape = (256, 256)  # Square matrix
            elif "bias" in param.name.lower() or "b" == param.name.lower():
                shape = (256,)

            inputs[param.name] = torch.randn(shape, dtype=torch.float32)

        return inputs

    @classmethod
    def parse_user_config(cls, config_str: Optional[str]) -> Dict[str, Any]:
        """Parse user configuration string.

        Format: "KEY1=VALUE1 KEY2=VALUE2"
        Example: "BLOCK_SIZE=512 num_warps=8"

        Args:
            config_str: Configuration string from CLI

        Returns:
            Dict of key-value pairs
        """
        config = {}
        if not config_str:
            return config

        for item in config_str.split():
            if "=" not in item:
                raise ValueError(f"Invalid config format: {item}. Expected KEY=VALUE")
            key, value = item.split("=", 1)
            try:
                # Try to parse as int first
                config[key] = int(value)
            except ValueError:
                try:
                    # Try to parse as float
                    config[key] = float(value)
                except ValueError:
                    # Keep as string
                    config[key] = value

        return config
