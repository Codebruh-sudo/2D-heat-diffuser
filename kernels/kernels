/*
 * diffusion_kernels.cu
 *
 * Pure CUDA C++ kernels for 2-D explicit heat diffusion (FTCS scheme).
 *
 * PDE:   ∂u/∂t = α ( ∂²u/∂x² + ∂²u/∂y² )
 *
 * Discretisation (5-point stencil, row-major u[i*nx + j]) 
 *
 *   u_new[i,j] = u[i,j]
 *              + rx * ( u[i][j-1] - 2·u[i][j] + u[i][j+1] )   ← x
 *              + ry * ( u[i-1][j] - 2·u[i][j] + u[i+1][j] )   ← y
 *
 *   rx = α·dt/dx²,   ry = α·dt/dy²
 *
 * Compile (inside Colab via nvcc):
 *   !nvcc -O3 --use_fast_math -Xcompiler -fPIC \
 *         -shared -o kernels/diffusion_kernels.so \
 *         kernels/diffusion_kernels.cu
 *
 * Four optimisation stages:
 *   1. diffuse_naive      - global memory baseline
 *   2. diffuse_shared      - shared memory tiling + halo
 *   3. diffuse_float4       - float4 vectorised loads
 *   4. diffuse_occupancy  - __launch_bounds__ occupancy tuning
 *
 * All kernels implement identical boundary conditions-
 *   Dirichlet  u = 0  on all four edges.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ═══════════════════════════════════════════════════════════════════
// Shared-memory tile size - used by kernels 2 and 4
// ═══════════════════════════════════════════════════════════════════
#define TILE     16
#define SMEM     (TILE + 2)     /* tile + 1-cell halo on each side */
#define SMEM_PAD (SMEM + 1)     /* +1 column padding to eliminate bank conflicts
                                   stride 19: gcd(19,32)=1 → 0 bank conflicts */

// ═══════════════════════════════════════════════════════════════════
// KERNEL 1 - Naive (global memory only)
//
// Every thread independently reads 5 floats from DRAM.
// No data reuse - each value is loaded ~5 times across the block.
//
// Arithmetic intensity:
//   8 FLOPs (4 adds, 4 mults) / 24 bytes (5R + 1W) = 0.33 FLOP/byte
//
// This is the baseline all other kernels are measured against.
// ═══════════════════════════════════════════════════════════════════
__global__ void diffuse_naive(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int   nx,
    int   ny,
    float rx,
    float ry)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   /* col (x) */
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   /* row (y) */

    /* Bounds guard — must come before any memory access */
    if (i >= ny || j >= nx) return;

    const int idx = i * nx + j;

    /* Use if-else instead of early return to keep warp threads converged.
       Early returns cause divergence: boundary threads exit while interior
       threads in the same warp continue, serialising execution. */
    if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1) {
        u_new[idx] = 0.0f;
    } else {
        const float center = u[idx];
        const float left   = u[idx - 1];           /* u[i  ][j-1] */
        const float right  = u[idx + 1];           /* u[i  ][j+1] */
        const float up     = u[(i - 1) * nx + j];  /* u[i-1][j  ] */
        const float down   = u[(i + 1) * nx + j];  /* u[i+1][j  ] */

        u_new[idx] = center
                   + rx * (left  - 2.0f * center + right)
                   + ry * (up    - 2.0f * center + down);
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL 2 — Shared Memory Tiling (bank-conflict-free, low divergence)
//
// Optimisations over the naive approach:
//   1. Shared memory tiling — neighbours served from on-chip SRAM
//   2. Padded column dimension (SMEM_PAD = 19) eliminates bank
//      conflicts. Original stride 18: gcd(18,32)=2 → 2-way conflicts.
//      Padded stride 19: gcd(19,32)=1 → 0 conflicts.
//   3. Cooperative linear tile load — ALL 256 threads participate in
//      loading the full 18×18 halo tile, replacing the old per-edge
//      if-branches (threadIdx.x==0, etc.) that caused warp divergence
//      in every warp (only 1-2 threads active per branch).
//   4. Unified write path — boundary threads write 0 via if-else
//      instead of early return, keeping warps converged.
//
// Shared memory per block: SMEM*SMEM_PAD*4 = 18*19*4 = 1368 bytes
// ═══════════════════════════════════════════════════════════════════
__global__ void diffuse_shared(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int   nx,
    int   ny,
    float rx,
    float ry)
{
    __shared__ float s[SMEM][SMEM_PAD];   /* padded to eliminate bank conflicts */

    /* Global indices */
    const int gi = blockIdx.y * TILE + threadIdx.y;
    const int gj = blockIdx.x * TILE + threadIdx.x;

    /* Shared-tile indices (offset +1 to leave room for halo) */
    const int li = threadIdx.y + 1;
    const int lj = threadIdx.x + 1;

    /* ── Cooperative tile load ─────────────────────────────────────
       All 256 threads load the full 18×18 halo tile linearly.
       This replaces per-edge if-branches that caused warp divergence
       (threadIdx.x==0 / ==TILE-1 meant only 1-2 threads per warp
       were active, serialising the other 30-31).
       With 324 elements and 256 threads: 2 iterations, divergence
       only in 1 warp on the second pass (68 active) vs. the old
       approach with divergence in ALL 8 warps.
       ────────────────────────────────────────────────────────────── */
    const int base_gi = blockIdx.y * TILE - 1;
    const int base_gj = blockIdx.x * TILE - 1;
    const int tid     = threadIdx.y * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < SMEM * SMEM; idx += blockDim.x * blockDim.y) {
        const int sr = idx / SMEM;
        const int sc = idx - sr * SMEM;
        const int gr = base_gi + sr;
        const int gc = base_gj + sc;
        s[sr][sc] = (gr >= 0 && gr < ny && gc >= 0 && gc < nx)
                    ? u[gr * nx + gc] : 0.0f;
    }

    /* ── Sync: all loads must finish before any reads ──────────── */
    __syncthreads();

    /* ── Unified write path: avoids divergent early returns ─────── */
    if (gi < ny && gj < nx) {
        float val = 0.0f;
        if (gi > 0 && gi < ny - 1 && gj > 0 && gj < nx - 1) {
            const float center = s[li  ][lj  ];
            const float left   = s[li  ][lj-1];
            const float right  = s[li  ][lj+1];
            const float up     = s[li-1][lj  ];
            const float down   = s[li+1][lj  ];
            val = center
                + rx * (left  - 2.0f * center + right)
                + ry * (up    - 2.0f * center + down);
        }
        u_new[gi * nx + gj] = val;
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL 3 — float4 Vectorised Loads
//
// Uses 128-bit (float4) load instructions to read 4 consecutive
// floats in a single memory transaction. The T4 memory controller
// issues 128-bit transactions natively, so a float4 load costs the
// same as a scalar float load but moves 4× the data.
//
// Each thread processes 4 consecutive j-columns (one float4 chunk).
// Constraint: nx must be a multiple of 4.
//
// Global thread layout:
//   threadIdx.x + blockIdx.x*blockDim.x  - chunk index (j/4)
//   threadIdx.y + blockIdx.y*blockDim.y  - row i
// ═══════════════════════════════════════════════════════════════════
__global__ void diffuse_float4(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int   nx,
    int   ny,
    float rx,
    float ry)
{
    /* j4 = first column of the 4-element chunk this thread owns */
    const int j4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int i  =  blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j4 + 3 >= nx) return;

    /* Boundary rows — zero all four outputs */
    if (i == 0 || i == ny - 1) {
        u_new[i*nx + j4    ] = 0.0f;
        u_new[i*nx + j4 + 1] = 0.0f;
        u_new[i*nx + j4 + 2] = 0.0f;
        u_new[i*nx + j4 + 3] = 0.0f;
        return;
    }

    /* ── 128-bit loads for current row and vertical neighbours ─── */
    const float4 row  = reinterpret_cast<const float4*>(u)[ i    * (nx/4) + j4/4 ];
    const float4 rowU = reinterpret_cast<const float4*>(u)[(i-1) * (nx/4) + j4/4 ];
    const float4 rowD = reinterpret_cast<const float4*>(u)[(i+1) * (nx/4) + j4/4 ];

    /* Scalar boundary neighbours at chunk edges */
    const float left0  = (j4 > 0)      ? u[i*nx + j4 - 1] : 0.0f;
    const float right3 = (j4+4 < nx)   ? u[i*nx + j4 + 4] : 0.0f;

    /* Unpack float4 into arrays for clean loop processing */
    const float vals[4] = { row.x,  row.y,  row.z,  row.w  };
    const float ups [4] = { rowU.x, rowU.y, rowU.z, rowU.w };
    const float dns [4] = { rowD.x, rowD.y, rowD.z, rowD.w };

    /* Build left/right neighbours within the chunk */
    const float lefts [4] = { left0,  row.x, row.y, row.z };
    const float rights[4] = { row.y,  row.z, row.w, right3 };

    /* Compute all 4 outputs */
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int col = j4 + k;
        if (col == 0 || col == nx - 1) {
            u_new[i*nx + col] = 0.0f;
        } else {
            u_new[i*nx + col] = vals[k]
                + rx * (lefts[k]  - 2.0f * vals[k] + rights[k])
                + ry * (ups[k]    - 2.0f * vals[k] + dns[k]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL 4 — Occupancy Tuned with __launch_bounds__
//
// Same optimised shared-memory algorithm as kernel 2 (padded SMEM,
// cooperative loading, unified write path), plus:
//
// __launch_bounds__(256, 4) tells nvcc to budget register allocation:
//   T4 (SM 7.5) has 65536 registers per SM.
//   256 threads/block × 4 blocks/SM → 64 regs/thread max budget.
//   Without the hint nvcc may spill to more registers → fewer active
//   blocks → lower occupancy → less latency hiding.
// ═══════════════════════════════════════════════════════════════════
__global__
__launch_bounds__(256, 4)   /* max 256 threads/block, aim ≥ 4 blocks/SM */
void diffuse_occupancy(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int   nx,
    int   ny,
    float rx,
    float ry)
{
    __shared__ float s[SMEM][SMEM_PAD];   /* padded to eliminate bank conflicts */

    const int gi = blockIdx.y * TILE + threadIdx.y;
    const int gj = blockIdx.x * TILE + threadIdx.x;
    const int li = threadIdx.y + 1;
    const int lj = threadIdx.x + 1;

    /* ── Cooperative tile load (same as kernel 2) ──────────────── */
    const int base_gi = blockIdx.y * TILE - 1;
    const int base_gj = blockIdx.x * TILE - 1;
    const int tid     = threadIdx.y * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < SMEM * SMEM; idx += blockDim.x * blockDim.y) {
        const int sr = idx / SMEM;
        const int sc = idx - sr * SMEM;
        const int gr = base_gi + sr;
        const int gc = base_gj + sc;
        s[sr][sc] = (gr >= 0 && gr < ny && gc >= 0 && gc < nx)
                    ? u[gr * nx + gc] : 0.0f;
    }

    __syncthreads();

    /* ── Unified write path ────────────────────────────────────── */
    if (gi < ny && gj < nx) {
        float val = 0.0f;
        if (gi > 0 && gi < ny - 1 && gj > 0 && gj < nx - 1) {
            const float center = s[li  ][lj  ];
            const float left   = s[li  ][lj-1];
            const float right  = s[li  ][lj+1];
            const float up     = s[li-1][lj  ];
            const float down   = s[li+1][lj  ];
            val = center
                + rx * (left  - 2.0f * center + right)
                + ry * (up    - 2.0f * center + down);
        }
        u_new[gi * nx + gj] = val;
    }
}
