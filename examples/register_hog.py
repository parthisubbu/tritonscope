import triton
import triton.language as tl
import torch

@triton.jit
def register_hog(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 8 independent accumulators, all live across the whole loop.
    # With BLOCK_SIZE=2048 and num_warps=4 (128 threads), each thread
    # holds 16 elements per vector -> 8 accumulators x 16 = 128+ live
    # registers per thread before temporaries.
    a0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a3 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a4 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a5 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a6 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    a7 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in range(8):
        x = tl.load(x_ptr + offsets)
        # exp/sin are opaque enough that the compiler can't fold
        # these into one expression; each accumulator stays live.
        a0 += tl.exp(x * 1.0001)
        a1 += tl.exp(x * 1.0002)
        a2 += tl.exp(x * 1.0003)
        a3 += tl.exp(x * 1.0004)
        a4 += tl.sin(x * 1.0005)
        a5 += tl.sin(x * 1.0006)
        a6 += tl.sin(x * 1.0007)
        a7 += tl.sin(x * 1.0008)

    tl.store(y_ptr + offsets, a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)

kernel_to_analyze = register_hog

def run_kernel(kernel_func, config):
    n_elements = 8192
    x = torch.randn(n_elements, device='cuda')
    y = torch.empty_like(x)
    block = config.get('BLOCK_SIZE', 2048)
    grid = (n_elements // block,)
    kernel_func[grid](
        x, y, n_elements,
        BLOCK_SIZE=block,
        num_warps=config.get('num_warps', 4),
        num_stages=config.get('num_stages', 3),
    )
