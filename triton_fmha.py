import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _triton_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Triton kernel for attention forward pass
    Simplified version for better stability
    """
    # Program IDs
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Q block index
    off_m = tl.program_id(1)
    
    # Initialize offsets
    offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * sm_scale).to(tl.float16)
    
    # Loop over K, V blocks
    for start_n in range(0, (off_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Update pointers
        k_ptrs = K + off_z * stride_kz + off_h * stride_kh + ((start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V + off_z * stride_vz + off_h * stride_vh + ((start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn)
        
        # Load k, v
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        
        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        
        # Apply causal mask
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Compute new m_i and l_i
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.exp(qk)
        
        # Apply mask for out-of-bounds
        p = tl.where((start_n + offs_n[None, :]) < N_CTX, p, 0.0)
        
        l_ij = tl.sum(p, 1)
        
        # Update accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    
    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def triton_attention(q, k, v, causal=False, dropout_p=0.0, sm_scale=None):
    """
    Triton implementation of attention mechanism
    
    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        v: Value tensor of shape (batch, seq_len, num_heads, head_dim)
        causal: Whether to apply causal masking
        dropout_p: Dropout probability (not implemented in this version)
        sm_scale: Scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim)
    """
    # Shape inference
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Validate shapes
    assert k.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v.shape == (batch_size, seq_len, num_heads, head_dim)
    
    # Set default scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Ensure tensors are contiguous and in correct format
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Output tensor
    o = torch.empty_like(q)
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid for launching kernels
    grid = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_M), 1)
    
    # Launch kernel
    _triton_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, num_heads, seq_len,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=head_dim,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=3,
    )
    
    return o


class TritonFusedMultiheadAttention(torch.nn.Module):
    """
    A fused multi-head attention module using Triton for optimization
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            torch.nn.init.constant_(self.q_proj.bias, 0.)
            torch.nn.init.constant_(self.k_proj.bias, 0.)
            torch.nn.init.constant_(self.v_proj.bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                need_weights: bool = False, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False, key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass using Triton-optimized attention
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            need_weights: Whether to return attention weights
            attn_mask: Attention mask (not used)
            is_causal: Whether to apply causal masking
            key_padding_mask: Key padding mask (not used)
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, tgt_len, embed_dim = query.size()
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Apply Triton attention
        attn_output = triton_attention(q, k, v, causal=is_causal, dropout_p=self.dropout)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, None
        else:
            return attn_output
