import torch
import torch.nn.functional as F
import time
import numpy as np
from fmha_operator import FusedMultiheadAttention
from triton_fmha import TritonFusedMultiheadAttention
import argparse


def benchmark_function(func, *args, warmup=3, repeat=10):
    """
    Benchmark a function and return execution times
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args)
    
    # Synchronize GPU if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual timing runs
    times = []
    for _ in range(repeat):
        start_time = time.time()
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.array(times), result


def create_test_tensors(batch_size, seq_len, embed_dim, num_heads, device='cpu'):
    """
    Create test tensors for benchmarking
    """
    head_dim = embed_dim // num_heads
    
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    key = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    value = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
    
    return query, key, value


def benchmark_pytorch_mha(batch_size, seq_len, embed_dim, num_heads, device='cpu'):
    """
    Benchmark PyTorch's MultiheadAttention
    """
    query, key, value = create_test_tensors(batch_size, seq_len, embed_dim, num_heads, device)
    
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)
    
    def run_mha():
        attn_output, _ = mha(query, key, value)
        return attn_output
    
    times, result = benchmark_function(run_mha)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape,
        'memory_usage': torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    }


def benchmark_custom_fmha(batch_size, seq_len, embed_dim, num_heads, device='cpu'):
    """
    Benchmark custom FMHA implementation
    """
    query, key, value = create_test_tensors(batch_size, seq_len, embed_dim, num_heads, device)
    
    fmha = FusedMultiheadAttention(embed_dim, num_heads, device=device)
    
    def run_fmha():
        attn_output = fmha(query, key, value, is_causal=False)
        return attn_output
    
    times, result = benchmark_function(run_fmha)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape,
        'memory_usage': torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    }


def benchmark_triton_fmha(batch_size, seq_len, embed_dim, num_heads, device='cpu'):
    """
    Benchmark Triton FMHA implementation
    """
    if not torch.cuda.is_available():
        print("Triton implementation requires CUDA, skipping...")
        return None
        
    query, key, value = create_test_tensors(batch_size, seq_len, embed_dim, num_heads, device)
    
    fmha = TritonFusedMultiheadAttention(embed_dim, num_heads, device=device)
    
    def run_triton_fmha():
        attn_output = fmha(query, key, value, is_causal=False)
        return attn_output
    
    times, result = benchmark_function(run_triton_fmha)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape,
        'memory_usage': torch.cuda.memory_allocated(device)
    }


def validate_results(batch_size, seq_len, embed_dim, num_heads, device='cpu'):
    """
    Validate that all implementations produce similar results
    """
    query, key, value = create_test_tensors(batch_size, seq_len, embed_dim, num_heads, device)
    
    # PyTorch MHA
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)
    with torch.no_grad():
        pytorch_out, _ = mha(query, key, value)
    
    # Custom FMHA
    fmha = FusedMultiheadAttention(embed_dim, num_heads, device=device)
    with torch.no_grad():
        custom_out = fmha(query, key, value, is_causal=False)
    
    # Check similarity
    diff = torch.abs(pytorch_out - custom_out).mean()
    max_diff = torch.abs(pytorch_out - custom_out).max()
    
    print(f"Validation Results:")
    print(f"  Mean absolute difference: {diff:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    if max_diff < 1e-3:
        print("  ✓ Results match within tolerance")
    else:
        print("  ✗ Results differ significantly")
    
    return diff, max_diff


def main():
    parser = argparse.ArgumentParser(description='Benchmark FMHA implementations')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run benchmarks on')
    
    args = parser.parse_args()
    
    print(f"Benchmarking on device: {args.device}")
    print(f"Parameters: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"embed_dim={args.embed_dim}, num_heads={args.num_heads}")
    print("-" * 80)
    
    # Validate results first
    validate_results(args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device)
    print("-" * 80)
    
    # Benchmark PyTorch MHA
    print("Benchmarking PyTorch MultiheadAttention...")
    pytorch_results = benchmark_pytorch_mha(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    
    # Benchmark custom FMHA
    print("Benchmarking Custom FMHA...")
    custom_results = benchmark_custom_fmha(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    
    # Benchmark Triton FMHA if available
    print("Benchmarking Triton FMHA...")
    triton_results = benchmark_triton_fmha(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    
    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"{'Implementation':<20} {'Mean Time (ms)':<15} {'Std Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    def ms_format(time_s):
        return f"{time_s*1000:.3f}"
    
    def mb_format(bytes_val):
        return f"{bytes_val / 1024 / 1024:.1f}" if bytes_val > 0 else "N/A"
    
    print(f"{'PyTorch MHA':<20} {ms_format(pytorch_results['mean_time']):<15} {ms_format(pytorch_results['std_time']):<15} {ms_format(pytorch_results['min_time']):<15} {ms_format(pytorch_results['max_time']):<15} {mb_format(pytorch_results['memory_usage']):<12}")
    print(f"{'Custom FMHA':<20} {ms_format(custom_results['mean_time']):<15} {ms_format(custom_results['std_time']):<15} {ms_format(custom_results['min_time']):<15} {ms_format(custom_results['max_time']):<15} {mb_format(custom_results['memory_usage']):<12}")
    
    if triton_results:
        print(f"{'Triton FMHA':<20} {ms_format(triton_results['mean_time']):<15} {ms_format(triton_results['std_time']):<15} {ms_format(triton_results['min_time']):<15} {ms_format(triton_results['max_time']):<15} {mb_format(triton_results['memory_usage']):<12}")
    else:
        print(f"{'Triton FMHA':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    pytorch_time = pytorch_results['mean_time']
    custom_time = custom_results['mean_time']
    
    speedup_custom = pytorch_time / custom_time
    print(f"Custom FMHA Speedup vs PyTorch: {speedup_custom:.2f}x")
    
    if triton_results:
        triton_time = triton_results['mean_time']
        speedup_triton = pytorch_time / triton_time
        print(f"Triton FMHA Speedup vs PyTorch: {speedup_triton:.2f}x")
    
    print("\nMemory Usage Analysis:")
    pytorch_mem = pytorch_results['memory_usage'] / (1024**2)
    custom_mem = custom_results['memory_usage'] / (1024**2)
    print(f"PyTorch MHA Memory: {pytorch_mem:.1f} MB")
    print(f"Custom FMHA Memory: {custom_mem:.1f} MB")
    
    if pytorch_mem > 0 and custom_mem > 0:
        mem_reduction = (pytorch_mem - custom_mem) / pytorch_mem * 100
        print(f"Memory Reduction: {mem_reduction:.1f}%")


if __name__ == "__main__":
    main()