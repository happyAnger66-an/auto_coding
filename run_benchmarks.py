#!/usr/bin/env python3
"""
Simple script to run FMHA benchmarks with common configurations
"""

import subprocess
import sys
import os


def run_benchmark(config_name, **kwargs):
    """
    Run benchmark with specified configuration
    """
    cmd = ["python", "benchmark_fmha.py"]
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"\nRunning {config_name} configuration...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/zhangxa/codes/auto_coding")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return False


def main():
    """
    Run benchmarks with different configurations
    """
    print("FMHA Benchmark Runner")
    print("=" * 50)
    
    # Small configuration (quick test)
    success_small = run_benchmark(
        "Small",
        batch_size=2,
        seq_len=128,
        embed_dim=256,
        num_heads=4
    )
    
    # Medium configuration
    success_medium = run_benchmark(
        "Medium", 
        batch_size=4,
        seq_len=512,
        embed_dim=512,
        num_heads=8
    )
    
    # Large configuration (if GPU available)
    if True:  # Always attempt, will handle GPU requirement internally
        success_large = run_benchmark(
            "Large",
            batch_size=8,
            seq_len=1024,
            embed_dim=1024,
            num_heads=16
        )
    
    print("\nBenchmark Summary:")
    print(f"Small config: {'✓' if success_small else '✗'}")
    print(f"Medium config: {'✓' if success_medium else '✗'}")
    print(f"Large config: {'✓' if 'success_large' in locals() and success_large else '?'}")


if __name__ == "__main__":
    main()