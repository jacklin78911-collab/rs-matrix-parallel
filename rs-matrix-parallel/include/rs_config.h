/*
 * Matrix-Parallel RS Encoding
 * Author: Liqian Lin
 * Date: 2026-02-01
 * Description: Configuration parameters for RS(n, k)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// #define DEBUG_MODE  // Uncomment to enable verbose logs

// Architecture tuning
// 16x16 blocks aligned with warp size (32) to minimize bank conflicts
#define TILE_DIM 16 
#define WARP_SIZE 32

// GF(2^8) specifics
#define GF_FIELD_SIZE 256
#define GF_ORDER 8

// Utility macros
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[ERROR] CUDA call failed: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
