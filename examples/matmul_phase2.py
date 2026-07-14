"""Matrix multiply kernel example for TrionScope Phase 2 testing."""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matrix multiply kernel: C = A @ B.

    Args:
        A: Input matrix (M x K)
        B: Input matrix (K x N)
        C: Output matrix (M x N)
        M, N, K: Matrix dimensions
        stride_*: Strides for each matrix
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes (constexpr)
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        # Load block of A
        A_ptrs = A + m_offsets * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load block of B
        B_ptrs = B + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    C_ptrs = C + m_offsets * stride_cm + n_offsets * stride_cn
    c_mask = (m_offsets < M) & (n_offsets < N)
    tl.store(C_ptrs, acc, mask=c_mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiply using Triton kernel.

    Args:
        A: Input matrix (M x K)
        B: Input matrix (K x N)

    Returns:
        Output matrix (M x N)
    """
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Ensure matrices are contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Launch kernel with 2D grid
    grid = (
        (M + 64 - 1) // 64,  # Number of blocks in M dimension
        (N + 64 - 1) // 64,  # Number of blocks in N dimension
    )

    # For analysis only (Phase 2 doesn't run kernels yet)
    return C


if __name__ == "__main__":
    print("MatMul kernel loaded. Use with: tritonscope analyze examples/matmul_phase2.py --kernel matmul_kernel")
