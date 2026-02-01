#include "rs_config.h"

// Stored in Constant Memory for broadcast efficiency
__constant__ unsigned short dc_log[2 * GF_FIELD_SIZE]; 
__constant__ unsigned char dc_exp[2 * GF_FIELD_SIZE];

void init_galois_tables(unsigned short* h_log, unsigned char* h_exp) {
    // Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    const int prim_poly = 0x11D;
    
    // Fill exp table
    int v = 1;
    for (int i = 0; i < GF_FIELD_SIZE - 1; ++i) {
        h_exp[i] = v;
        h_log[v] = i;
        v <<= 1;
        if (v & 0x100) v ^= prim_poly;
    }
    
    // Optimization: Extended tables to avoid modulo in kernel
    // Duplicating the table allows us to do: exp[log[a] + log[b]] directly
    for (int i = GF_FIELD_SIZE - 1; i < 2 * GF_FIELD_SIZE; ++i) {
        h_exp[i] = h_exp[i - (GF_FIELD_SIZE - 1)];
    }
    
    // Padding logic to prevent out-of-bound access
    for (int i = 0; i < GF_FIELD_SIZE; ++i) {
        h_log[i + GF_FIELD_SIZE] = h_log[i]; 
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(dc_log, h_log, 2 * GF_FIELD_SIZE * sizeof(unsigned short)));
    CUDA_CHECK(cudaMemcpyToSymbol(dc_exp, h_exp, 2 * GF_FIELD_SIZE * sizeof(unsigned char)));
}
