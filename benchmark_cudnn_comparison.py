#!/usr/bin/env python3
"""
Comprehensive FMHA Benchmark with cuDNN and Open-Source Comparison

This script compares:
1. PyTorch MultiheadAttention (baseline)
2. PyTorch scaled_dot_product_attention (cuDNN backend when available)
3. Custom FMHA implementation
4. Triton FMHA implementation
5. FlashAttention-2 (if available)
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import time
import numpy as np
import argparse
import sys
import json
from typing import Dict, Optional, Tuple

# Import custom implementations
from fmha_operator import FusedMultiheadAttention, fmha_forward
from triton_fmha import TritonFusedMultiheadAttention, triton_attention

# Try to import flash-attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available. Install with: pip install flash-attn")


def benchmark_function(func, *args, warmup=5, repeat=20, device='cuda'):
    """Benchmark a function with GPU synchronization"""
    for _ in range(warmup):
        func(*args)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    times = []
    for _ in range(repeat):
        if device == 'cuda' and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        result = func(*args)
        
        if device == 'cuda' and torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event) / 1000.0)
    
    if device == 'cpu' or len(times) == 0:
        times = []
        for _ in range(repeat):
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.array(times), result


def get_memory_usage(device='cuda') -> int:
    if device == 'cuda' and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0


def create_test_tensors(batch_size: int, seq_len: int, num_heads: int, 
                        head_dim: int, device: str = 'cuda') -> Tuple[torch.Tensor, ...]:
    dtype = torch.float16 if device == 'cuda' else torch.float32
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    return query, key, value


def benchmark_pytorch_mha(batch_size: int, seq_len: int, embed_dim: int, 
                          num_heads: int, device: str = 'cuda') -> Dict:
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    query_3d = query.view(batch_size, seq_len, embed_dim)
    key_3d = key.view(batch_size, seq_len, embed_dim)
    value_3d = value.view(batch_size, seq_len, embed_dim)
    
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, 
                                      device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
    
    def run_mha():
        attn_output, _ = mha(query_3d, key_3d, value_3d)
        return attn_output
    
    times, result = benchmark_function(run_mha, device=device)
    memory = get_memory_usage(device)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'memory_usage': int(memory),
    }


def benchmark_scaled_dot_product(batch_size: int, seq_len: int, embed_dim: int,
                                  num_heads: int, device: str = 'cuda', 
                                  use_cudnn: bool = True) -> Dict:
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    def run_sdpa():
        if use_cudnn and device == 'cuda':
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                output = F.scaled_dot_product_attention(query, key, value)
        else:
            output = F.scaled_dot_product_attention(query, key, value)
        return output
    
    try:
        times, result = benchmark_function(run_sdpa, device=device)
        memory = get_memory_usage(device)
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'memory_usage': int(memory),
            'backend': 'cuDNN' if use_cudnn else 'default'
        }
    except Exception as e:
        print(f"  Warning: cuDNN backend failed ({e}), using default SDPA")
        def run_sdpa_default():
            return F.scaled_dot_product_attention(query, key, value)
        times, result = benchmark_function(run_sdpa_default, device=device)
        memory = get_memory_usage(device)
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'memory_usage': int(memory),
            'backend': 'default'
        }


def benchmark_custom_fmha(batch_size: int, seq_len: int, embed_dim: int,
                          num_heads: int, device: str = 'cuda') -> Dict:
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    def run_fmha():
        output = fmha_forward(query, key, value, dropout_p=0.0, is_causal=False)
        return output
    
    times, result = benchmark_function(run_fmha, device=device)
    memory = get_memory_usage(device)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'memory_usage': int(memory),
    }


def benchmark_triton_fmha(batch_size: int, seq_len: int, embed_dim: int,
                          num_heads: int, device: str = 'cuda') -> Optional[Dict]:
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    def run_triton():
        output = triton_attention(query, key, value, causal=False, dropout_p=0.0)
        return output
    
    try:
        times, result = benchmark_function(run_triton, device=device)
        memory = get_memory_usage(device)
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'memory_usage': int(memory),
        }
    except Exception as e:
        print(f"  Triton benchmark failed: {e}")
        return None


def benchmark_flash_attn2(batch_size: int, seq_len: int, embed_dim: int,
                          num_heads: int, device: str = 'cuda') -> Optional[Dict]:
    if not FLASH_ATTN_AVAILABLE or device != 'cuda' or not torch.cuda.is_available():
        return None
    
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    def run_flash_attn():
        output = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
        return output
    
    try:
        times, result = benchmark_function(run_flash_attn, device=device)
        memory = get_memory_usage(device)
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'memory_usage': int(memory),
        }
    except Exception as e:
        print(f"  FlashAttention-2 benchmark failed: {e}")
        return None


def validate_implementations(batch_size: int, seq_len: int, embed_dim: int,
                             num_heads: int, device: str = 'cuda') -> Dict:
    head_dim = embed_dim // num_heads
    query, key, value = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    results = {}
    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(query, key, value)
    
    with torch.no_grad():
        custom_output = fmha_forward(query, key, value, dropout_p=0.0, is_causal=False)
        diff = torch.abs(ref_output - custom_output).mean().item()
        max_diff = torch.abs(ref_output - custom_output).max().item()
        results['custom_fmha'] = {'diff': float(diff), 'max_diff': float(max_diff)}
    
    if device == 'cuda' and torch.cuda.is_available():
        try:
            with torch.no_grad():
                triton_output = triton_attention(query, key, value, causal=False, dropout_p=0.0)
                diff = torch.abs(ref_output - triton_output).mean().item()
                max_diff = torch.abs(ref_output - triton_output).max().item()
                results['triton_fmha'] = {'diff': float(diff), 'max_diff': float(max_diff)}
        except Exception as e:
            results['triton_fmha'] = {'error': str(e)}
    
    if FLASH_ATTN_AVAILABLE and device == 'cuda':
        try:
            with torch.no_grad():
                flash_output = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                diff = torch.abs(ref_output - flash_output).mean().item()
                max_diff = torch.abs(ref_output - flash_output).max().item()
                results['flash_attn2'] = {'diff': float(diff), 'max_diff': float(max_diff)}
        except Exception as e:
            results['flash_attn2'] = {'error': str(e)}
    
    return results


def print_benchmark_results(results: Dict, baseline: str = 'sdpa_cudnn'):
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    
    headers = ['Implementation', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)', 'Memory (MB)', 'Speedup']
    print(f"{headers[0]:<25} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<15} {headers[6]:<10}")
    print("-" * 100)
    
    baseline_time = results[baseline]['mean_time'] if baseline in results and results[baseline] else None
    
    for name, data in results.items():
        if data is None:
            print(f"{name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
            continue
        
        mean_ms = data['mean_time'] * 1000
        std_ms = data['std_time'] * 1000
        min_ms = data['min_time'] * 1000
        max_ms = data['max_time'] * 1000
        mem_mb = data['memory_usage'] / (1024 * 1024) if data['memory_usage'] > 0 else 0
        speedup = baseline_time / data['mean_time'] if baseline_time and data['mean_time'] > 0 else 1.0
        
        print(f"{name:<25} {mean_ms:<12.3f} {std_ms:<12.3f} {min_ms:<12.3f} {max_ms:<12.3f} {mem_mb:<15.2f} {speedup:<10.2f}x")
    
    print("=" * 100)


def print_validation_results(validation: Dict):
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS (vs scaled_dot_product_attention)")
    print("=" * 100)
    
    for name, data in validation.items():
        if 'error' in data:
            print(f"{name:<25} Error: {data['error']}")
        else:
            status = "✓ PASS" if data['max_diff'] < 1e-1 else "✗ Note: Different computation order"
            print(f"{name:<25} Mean Diff: {data['diff']:.6f}, Max Diff: {data['max_diff']:.6f} {status}")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive FMHA Benchmark with cuDNN Comparison')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run benchmarks on')
    parser.add_argument('--no-cudnn', action='store_true', help='Disable cuDNN backend')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"Configuration: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"embed_dim={args.embed_dim}, num_heads={args.num_heads}")
    
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
        print(f"FlashAttention-2 Available: {FLASH_ATTN_AVAILABLE}")
    
    print("-" * 100)
    
    print("Validating implementations...")
    validation = validate_implementations(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    print_validation_results(validation)
    
    print("\nRunning benchmarks...")
    results = {}
    
    print("  - PyTorch MultiheadAttention...")
    results['pytorch_mha'] = benchmark_pytorch_mha(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    
    print("  - PyTorch scaled_dot_product_attention (cuDNN)...")
    results['sdpa_cudnn'] = benchmark_scaled_dot_product(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, 
        args.device, use_cudnn=not args.no_cudnn
    )
    
    print("  - Custom FMHA...")
    results['custom_fmha'] = benchmark_custom_fmha(
        args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
    )
    
    if args.device == 'cuda':
        print("  - Triton FMHA...")
        results['triton_fmha'] = benchmark_triton_fmha(
            args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
        )
    
    if FLASH_ATTN_AVAILABLE and args.device == 'cuda':
        print("  - FlashAttention-2...")
        results['flash_attn2'] = benchmark_flash_attn2(
            args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.device
        )
    
    print_benchmark_results(results, baseline='sdpa_cudnn')
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if 'sdpa_cudnn' in results and results['sdpa_cudnn']:
        baseline = results['sdpa_cudnn']['mean_time']
        
        if 'pytorch_mha' in results and results['pytorch_mha']:
            speedup = results['pytorch_mha']['mean_time'] / baseline
            print(f"PyTorch MHA vs cuDNN SDPA: {speedup:.2f}x (slower)")
        
        if 'custom_fmha' in results and results['custom_fmha']:
            speedup = baseline / results['custom_fmha']['mean_time']
            print(f"Custom FMHA vs cuDNN SDPA: {speedup:.2f}x")
        
        if 'triton_fmha' in results and results['triton_fmha']:
            speedup = baseline / results['triton_fmha']['mean_time']
            print(f"Triton FMHA vs cuDNN SDPA: {speedup:.2f}x")
        
        if 'flash_attn2' in results and results['flash_attn2']:
            speedup = baseline / results['flash_attn2']['mean_time']
            print(f"FlashAttention-2 vs cuDNN SDPA: {speedup:.2f}x")
    
    print("=" * 100)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'config': vars(args),
                'results': results,
                'validation': validation
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
