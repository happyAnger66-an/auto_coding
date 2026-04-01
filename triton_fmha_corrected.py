"""
Corrected Triton implementation of Flash Multi-Head Attention (FMHA)
This implementation fixes indexing issues in the previous version
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_forward(
    Q, K, V, 
    O, 
    L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,
    stride_mz, stride_mh, stride_mm,
    Z, H, N_CTX, 
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Corrected forward pass for Flash Attention using Triton
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over blocks of sequence dimension
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K and V for this block
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vk, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)

        # Compute Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float32), 
                     tl.trans(k).to(tl.float32), 
                     allow_tf32=True)
        
        # Apply scale
        qk *= sm_scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Compute new m and l
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update accumulator
        alpha = tl.math.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v, allow_tf32=True)
        
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # Advance K and V pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(O.type.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # Store L and M values for backward pass
    l_ptrs = L + off_hz * stride_lh + offs_m * stride_lm
    m_ptrs = M + off_hz * stride_mh + offs_m * stride_mm
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_triton(q, k, v, causal=False, sm_scale=None):
    """
    Flash Attention implementation using Triton
    """
    # Verify constraints
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Query, Key, Value must have same head dimension"
    assert q.shape[-2] == k.shape[-2] == v.shape[-2], "Q, K, V must have same sequence length"
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    
    # Handle batch dimensions
    if len(q.shape) == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    
    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Validate head dimension
    assert head_dim <= 256, "Head dimension must be <= 256 for this implementation"
    assert head_dim % 8 == 0, "Head dimension must be divisible by 8"
    
    # Output tensor
    o = torch.empty_like(q)
    
    # Additional tensors for normalization
    L = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
    M = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
    
    # Grid for Triton kernel launch
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_M']),
        batch_size * num_heads,
        1
    )
    
    # Launch kernel
    _flash_attn_forward[grid](
        q, k, v, o, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        batch_size, num_heads, seq_len, sm_scale,
        BLOCK_M=128, 
        BLOCK_N=64, 
        BLOCK_DMODEL=head_dim,
        IS_CAUSAL=causal,
        num_stages=1,
        num_warps=4
    )
    
    return o.squeeze(0) if len(o.shape) == 3 else o


class TritonFlashMHA(torch.nn.Module):
    """
    Flash Multi-Head Attention using Triton kernels
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear layers for Q, K, V projections
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, 
                need_weights=False, average_attn_weights=True, is_causal=False):
        """
        Forward pass using Triton-based Flash Attention
        """
        if not self.batch_first:
            # Convert to batch_first format
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
            
        B, L, E = query.shape  # Batch, Length, Embedding
        _, S, _ = key.shape    # Sequence length of key/value
        
        # Project Q, K, V
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Flash Attention using Triton
        output = flash_attention_triton(q, k, v, causal=is_causal, sm_scale=self.scale)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, L, E)  # [B, L, E]
        output = self.out_proj(output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
            
        if need_weights:
            # For Triton implementation, returning attention weights is complex
            # We return None to indicate weights are not available
            return output, None
        else:
            return output