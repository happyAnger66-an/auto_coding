# Flash Multi-Head Attention (FMHA) Implementation

This repository contains implementations of Flash Multi-Head Attention operators with performance comparisons against standard PyTorch implementations.

## Overview

Flash Multi-Head Attention (FMHA) is an optimized attention mechanism that reduces memory usage and computational complexity compared to traditional attention mechanisms. This implementation includes:

1. A custom PyTorch implementation of FMHA
2. A Triton-optimized CUDA kernel implementation
3. Comprehensive benchmarking tools for performance comparison

## Features

- Memory-efficient attention computation
- Support for causal masking
- Compatible with PyTorch's autograd system
- Triton-optimized kernels for GPU acceleration
- Comprehensive benchmarking framework

## Files

- `fmha_operator.py`: Custom PyTorch implementation of FMHA
- `triton_fmha.py`: Triton-optimized attention kernels
- `benchmark_fmha.py`: Performance comparison tool
- `test_fmha.py`: Basic functionality tests
- `requirements.txt`: Project dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

```bash
python test_fmha.py
```

### Running Benchmarks

```bash
python benchmark_fmha.py --batch-size 4 --seq-len 512 --embed-dim 512 --num-heads 8
```

Available options:
- `--batch-size`: Batch size (default: 4)
- `--seq-len`: Sequence length (default: 512)
- `--embed-dim`: Embedding dimension (default: 512)
- `--num-heads`: Number of attention heads (default: 8)
- `--device`: Device to run benchmarks on (default: cuda if available, otherwise cpu)

## Performance Comparison

The benchmarking tool compares:
- PyTorch's built-in MultiheadAttention
- Custom FMHA implementation
- Triton-optimized FMHA implementation (GPU only)

Metrics measured:
- Execution time (mean, std, min, max)
- Memory usage
- Speedup ratios

## Architecture

### Custom FMHA Implementation
The custom implementation follows the FlashAttention principles:
- Chunked computation to reduce memory requirements
- Fused operations to minimize data movement
- Online softmax computation for numerical stability

### Triton Implementation
The Triton implementation provides:
- Low-level CUDA kernel optimizations
- Efficient memory access patterns
- Better utilization of GPU resources

## References

This implementation is inspired by the FlashAttention paper and optimized for practical usage scenarios.