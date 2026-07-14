import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_scaled_robust(x_ptr, scale, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    m = float('-inf')
    d = 0.0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_cols)
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        x_scaled = x * scale
        m_block = tl.max(x_scaled, axis=0)
        m_old = m
        m = tl.maximum(m, m_block)
        d = d * tl.exp(m_old - m)
        exp_x = tl.exp(x_scaled - m)
        d = d + tl.sum(exp_x * mask)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        x_scaled = x * scale
        exp_x = tl.exp(x_scaled - m)
        softmax_x = exp_x / d
        tl.store(output_ptr + row_start + offsets, softmax_x, mask=mask)

kernel_to_analyze = fused_softmax_scaled_robust

def run_kernel(kernel_func, config):
    """Generic runner for softmax kernel"""
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    scale = 1.0 / (seq_len ** 0.5)
    
    grid = (batch_size,)
    # BUG FIX: pass num_warps and num_stages through to the launch,
    # otherwise --config values were silently ignored.
    kernel_func[grid](
        x, scale, y, seq_len,
        BLOCK_SIZE=config.get('BLOCK_SIZE', 256),
        num_warps=config.get('num_warps', 4),
        num_stages=config.get('num_stages', 3),
    )
    torch.cuda.synchronize()
