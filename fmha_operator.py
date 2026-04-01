import torch
import torch.nn.functional as F
from typing import Optional


def fmha_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                 dropout_p: float = 0.0, is_causal: bool = False,
                 scale: Optional[float] = None) -> torch.Tensor:
    """
    Flash Multi-Head Attention forward pass implementation
    
    This implementation follows the same computation as F.scaled_dot_product_attention
    but with a memory-efficient approach.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len_q, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len_k, num_heads, head_dim)
        v: Value tensor of shape (batch_size, seq_len_k, num_heads, head_dim)
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Scaling factor for attention scores
        
    Returns:
        Output tensor of shape (batch_size, seq_len_q, num_heads, head_dim)
    """
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Compute attention scores: (batch, heads, seq_q, seq_k)
    # Transpose to get (batch, heads, seq, head_dim) for bmm
    q_transposed = q.transpose(1, 2)  # (batch, heads, seq_q, head_dim)
    k_transposed = k.transpose(1, 2)  # (batch, heads, seq_k, head_dim)
    v_transposed = v.transpose(1, 2)  # (batch, heads, seq_k, head_dim)
    
    # Reshape for batch matrix multiply
    # (batch * heads, seq_q, head_dim)
    q_reshaped = q_transposed.reshape(batch_size * num_heads, seq_len_q, head_dim)
    k_reshaped = k_transposed.reshape(batch_size * num_heads, seq_len_k, head_dim)
    v_reshaped = v_transposed.reshape(batch_size * num_heads, seq_len_k, head_dim)
    
    # Compute attention scores: (batch * heads, seq_q, seq_k)
    attn_weights = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device), 
            diagonal=1
        )
        attn_weights.masked_fill_(causal_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Apply dropout if needed
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Compute output: (batch * heads, seq_q, head_dim)
    output = torch.bmm(attn_weights, v_reshaped)
    
    # Reshape back to (batch, heads, seq_q, head_dim)
    output = output.view(batch_size, num_heads, seq_len_q, head_dim)
    
    # Transpose back to (batch, seq_q, heads, head_dim)
    output = output.transpose(1, 2).contiguous()
    
    return output


class FusedMultiheadAttention(torch.nn.Module):
    """
    A fused multi-head attention module that implements Flash Multi-Head Attention
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
        Forward pass of the fused multi-head attention
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            is_causal: Whether to apply causal masking
            key_padding_mask: Key padding mask
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, tgt_len, embed_dim = query.size()
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Apply flash attention
        attn_output = fmha_forward(q, k, v, self.dropout, is_causal=is_causal)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, None
        else:
            return attn_output


def fmha_forward_3d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    embed_dim: int, num_heads: int,
                    dropout_p: float = 0.0, is_causal: bool = False,
                    scale: Optional[float] = None) -> torch.Tensor:
    """
    FMHA forward pass with 3D input tensors (batch, seq, embed_dim)
    
    This is a convenience function that handles the reshaping internally.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len_q, embed_dim)
        key: Key tensor of shape (batch_size, seq_len_k, embed_dim)
        value: Value tensor of shape (batch_size, seq_len_k, embed_dim)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Scaling factor for attention scores
        
    Returns:
        Output tensor of shape (batch_size, seq_len_q, embed_dim)
    """
    batch_size, seq_len_q, _ = query.shape
    _, seq_len_k, _ = key.shape
    head_dim = embed_dim // num_heads
    
    # Reshape to 4D: (batch, seq, heads, head_dim)
    q = query.view(batch_size, seq_len_q, num_heads, head_dim)
    k = key.view(batch_size, seq_len_k, num_heads, head_dim)
    v = value.view(batch_size, seq_len_k, num_heads, head_dim)
    
    # Apply FMHA
    output = fmha_forward(q, k, v, dropout_p, is_causal, scale)
    
    # Reshape back to 3D
    output = output.view(batch_size, seq_len_q, embed_dim)
    
    return output
