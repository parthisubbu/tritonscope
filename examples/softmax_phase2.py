"""Softmax kernel example for TrionScope Phase 2 testing."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
    """Softmax kernel with block-wise computation.

    Args:
        X: Input tensor
        Y: Output tensor
        stride_x: Stride for input
        stride_y: Stride for output
        N: Number of elements
        BLOCK_SIZE: Block size (constexpr)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load a block of data
    X_ptrs = X + row_idx * stride_x + col_offsets
    x = tl.load(X_ptrs, mask=mask, other=float('-inf'))

    # Compute softmax (log-sum-exp trick for numerical stability)
    x_max = tl.max(x, axis=0)
    x_safe = x - x_max
    exp_x = tl.exp(x_safe)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp

    # Store results
    Y_ptrs = Y + row_idx * stride_y + col_offsets
    tl.store(Y_ptrs, y, mask=mask)


def softmax(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax using Triton kernel.

    Args:
        X: Input tensor
        dim: Dimension to apply softmax

    Returns:
        Output tensor with same shape as input
    """
    if dim != -1:
        X = X.transpose(-1, dim)

    batch_size, seq_len = X.shape
    Y = torch.empty_like(X)

    # Use triton.testing.do_bench for consistent benchmarking
    grid = (batch_size,)

    def _kernel():
        softmax_kernel[grid](
            X.contiguous(),
            Y.contiguous(),
            X.stride(0),
            Y.stride(0),
            seq_len,
            BLOCK_SIZE=256,
            num_warps=4,
            num_stages=3,
        )

    # For analysis only (Phase 2 doesn't run kernels yet)
    return Y


if __name__ == "__main__":
    # Example usage (requires Phase 2 to run)
    print("Softmax kernel loaded. Use with: tritonscope analyze examples/softmax_phase2.py --kernel softmax_kernel")
