# Matrix-Parallel RS Encoding

High-performance implementation of Reed-Solomon encoding using Hierarchical Blocking strategy on NVIDIA GPUs.
Source code for the paper: **"Matrix-Parallel Framework for Reed-Solomon Codes: Accelerating Encoding via Hierarchical Blocking Strategy"**.

## Overview

This project implements a dense matrix multiplication approach to RS encoding over GF(2^8). Unlike traditional polynomial-based methods (e.g., ISA-L), this implementation utilizes the GPU's memory hierarchy (Shared Memory/L1) to maximize bandwidth utilization.

**Key Optimizations:**
*   **Hierarchical Blocking**: Tiled matrix multiplication ($16 \times 16$) to fit Shared Memory.
*   **Modulo-Free Arithmetic**: Extended 510-entry log tables to remove expensive integer division.
*   **Coalesced Access**: Memory layout optimization for 128-byte transactions.

## Environment

*   **GPU**: NVIDIA A100 / RTX 4090 / V100
*   **CUDA**: 11.x or higher
*   **OS**: Linux (Tested on Ubuntu 22.04)

## Build

Using CMake:

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

Run the benchmark:

```bash
./rs_benchmark [n] [k]
# Example: RS(255, 223)
./rs_benchmark 255 223
```

## Performance Data

Evaluated on NVIDIA A100 (40GB):

| Metric | Value | Note |
| --- | --- | --- |
| Peak Throughput | 40.5 GB/s | Batch size = 1GB |
| Speedup vs CPU | 4.1x | vs Intel Xeon Gold (ISA-L) |
| Energy Efficiency | 8.7 GB/J | Measured via nvidia-smi |

## License

MIT License.
