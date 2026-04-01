"""
Test script for the corrected FMHA implementations
"""

import torch
from fmha_operator import FlashMHA
from triton_fmha_corrected import TritonFlashMHA

def test_corrected_implementations():
    """Test the corrected implementations"""
    print("Testing corrected implementations...")
    
    # Parameters
    batch_size = 2
    seq_len = 64
    embed_dim = 128
    num_heads = 4
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    
    # Test FlashMHA
    print("Testing FlashMHA...")
    flash_mha = FlashMHA(embed_dim, num_heads).cuda()
    output_flash = flash_mha(x, x, x)
    print(f"FlashMHA output shape: {output_flash.shape}")
    
    # Test Corrected TritonFlashMHA
    print("Testing Corrected TritonFlashMHA...")
    try:
        triton_mha = TritonFlashMHA(embed_dim, num_heads).cuda()
        output_triton = triton_mha(x, x, x)
        print(f"Corrected TritonFlashMHA output shape: {output_triton.shape}")
    except Exception as e:
        print(f"Corrected TritonFlashMHA failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with standard PyTorch MHA for comparison
    print("Testing standard PyTorch MHA...")
    std_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    output_std, _ = std_mha(x, x, x)
    print(f"Standard MHA output shape: {output_std.shape}")
    
    print("Test completed!")


def test_accuracy():
    """Test accuracy between implementations"""
    print("\nTesting accuracy between implementations...")
    
    batch_size = 1
    seq_len = 32
    embed_dim = 64
    num_heads = 2
    
    # Use a fixed seed for reproducible results
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    
    # Get outputs from different implementations
    flash_mha = FlashMHA(embed_dim, num_heads).cuda()
    std_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    
    with torch.no_grad():
        output_flash = flash_mha(x, x, x)
        output_std, _ = std_mha(x, x, x)
        
        # Calculate differences
        diff_mean = torch.mean(torch.abs(output_flash - output_std)).item()
        diff_max = torch.max(torch.abs(output_flash - output_std)).item()
        
        print(f"Mean absolute difference: {diff_mean:.6f}")
        print(f"Max absolute difference: {diff_max:.6f}")
        
        # They should be reasonably close
        if diff_mean < 0.01:
            print("✓ Implementations produce similar results")
        else:
            print("⚠ Implementations differ significantly")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Testing on GPU.")
        torch.cuda.empty_cache()
        test_corrected_implementations()
        test_accuracy()
    else:
        print("CUDA not available. Please install PyTorch with CUDA support.")