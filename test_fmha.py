import torch
from fmha_operator import FusedMultiheadAttention
from triton_fmha import TritonFusedMultiheadAttention


def test_custom_fmha():
    """
    Test the custom FMHA implementation
    """
    print("Testing Custom FMHA Implementation...")
    
    # Parameters
    batch_size = 2
    seq_len = 16
    embed_dim = 128
    num_heads = 4
    
    # Create test inputs
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize model
    fmha = FusedMultiheadAttention(embed_dim, num_heads)
    
    # Forward pass
    output = fmha(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Custom FMHA test passed\n")


def test_triton_fmha():
    """
    Test the Triton FMHA implementation
    """
    print("Testing Triton FMHA Implementation...")
    
    # Only run if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton test")
        return
    
    # Parameters
    batch_size = 2
    seq_len = 16
    embed_dim = 128
    num_heads = 4
    
    # Move to GPU
    device = torch.device('cuda')
    query = torch.randn(batch_size, seq_len, embed_dim).to(device)
    key = torch.randn(batch_size, seq_len, embed_dim).to(device)
    value = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    # Initialize model
    fmha = TritonFusedMultiheadAttention(embed_dim, num_heads).to(device)
    
    # Forward pass
    output = fmha(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Triton FMHA test passed\n")


def test_comparison_with_pytorch():
    """
    Compare custom implementation with PyTorch's MultiheadAttention
    """
    print("Comparing with PyTorch MultiheadAttention...")
    
    # Parameters
    batch_size = 2
    seq_len = 16
    embed_dim = 128
    num_heads = 4
    
    # Create identical inputs
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    
    # PyTorch implementation
    pytorch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    pytorch_out, _ = pytorch_mha(query, key, value)
    
    # Custom implementation
    custom_mha = FusedMultiheadAttention(embed_dim, num_heads)
    # Need to copy weights to ensure same initialization
    with torch.no_grad():
        custom_mha.q_proj.weight.copy_(pytorch_mha.in_proj_weight[:embed_dim])
        custom_mha.k_proj.weight.copy_(pytorch_mha.in_proj_weight[embed_dim:2*embed_dim])
        custom_mha.v_proj.weight.copy_(pytorch_mha.in_proj_weight[2*embed_dim:])
        custom_mha.q_proj.bias.copy_(pytorch_mha.in_proj_bias[:embed_dim])
        custom_mha.k_proj.bias.copy_(pytorch_mha.in_proj_bias[embed_dim:2*embed_dim])
        custom_mha.v_proj.bias.copy_(pytorch_mha.in_proj_bias[2*embed_dim:])
        custom_mha.out_proj.weight.copy_(pytorch_mha.out_proj.weight)
        custom_mha.out_proj.bias.copy_(pytorch_mha.out_proj.bias)
    
    custom_out = custom_mha(query, key, value)
    
    # Calculate difference
    diff = torch.abs(pytorch_out - custom_out).mean()
    max_diff = torch.abs(pytorch_out - custom_out).max()
    
    print(f"Mean absolute difference: {diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print("✓ Comparison test completed\n")


if __name__ == "__main__":
    test_custom_fmha()
    test_triton_fmha()
    test_comparison_with_pytorch()
    print("All tests completed!")