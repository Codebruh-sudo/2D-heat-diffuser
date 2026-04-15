import subprocess, os

os.makedirs("kernels",     exist_ok=True)
os.makedirs("benchmarks",  exist_ok=True)

result = subprocess.run(["pip", "install", "pycuda", "--quiet"],
                        capture_output=True, text=True)
print("pycuda installed.")
print(f"GPU: {subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}")

# ═══════════════════════════════════════════════════════════════════
# CELL 2 — Write the .cu file to disk
#
# In a real project you would git clone this file.
# In Colab we write it programmatically so the notebook is
# self-contained. The content is identical to diffusion_kernels.cu.
# ═══════════════════════════════════════════════════════════════════
cu_source = r"""
/*
 * kernels/diffusion_kernels.cu
 *
 * Pure CUDA C++ — 2D heat diffusion, 4 optimisation stages.
 * Compiled with nvcc and loaded by PyCUDA at runtime.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define TILE     16
#define SMEM     (TILE + 2)
#define SMEM_PAD (SMEM + 1)     /* +1 padding to eliminate bank conflicts */

/* ----------------------------------------------------------------
 * KERNEL 1 — Naive (global memory baseline)
 * AI = 8 FLOP / 24 bytes = 0.33 FLOP/byte
 * ---------------------------------------------------------------- */
__global__ void diffuse_naive(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int nx, int ny, float rx, float ry)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= ny || j >= nx) return;

    const int idx = i*nx + j;

    /* if-else instead of early return to keep warp threads converged */
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        u_new[idx] = 0.0f;
    } else {
        const float center = u[idx];
        const float left   = u[idx - 1];
        const float right  = u[idx + 1];
        const float up     = u[(i-1)*nx + j];
        const float down   = u[(i+1)*nx + j];

        u_new[idx] = center
                   + rx * (left  - 2.0f*center + right)
                   + ry * (up    - 2.0f*center + down);
    }
}

/* ----------------------------------------------------------------
 * KERNEL 2 — Shared Memory Tiling (bank-conflict-free, low divergence)
 * Padded SMEM_PAD=19 eliminates bank conflicts (gcd(19,32)=1).
 * Cooperative linear load replaces per-edge if-branches to reduce
 * warp divergence. Unified write path avoids divergent early returns.
 * ---------------------------------------------------------------- */
__global__ void diffuse_shared(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int nx, int ny, float rx, float ry)
{
    __shared__ float s[SMEM][SMEM_PAD];

    const int gi = blockIdx.y * TILE + threadIdx.y;
    const int gj = blockIdx.x * TILE + threadIdx.x;
    const int li = threadIdx.y + 1;
    const int lj = threadIdx.x + 1;

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

    if (gi < ny && gj < nx) {
        float val = 0.0f;
        if (gi > 0 && gi < ny-1 && gj > 0 && gj < nx-1) {
            const float center = s[li  ][lj  ];
            const float left   = s[li  ][lj-1];
            const float right  = s[li  ][lj+1];
            const float up     = s[li-1][lj  ];
            const float down   = s[li+1][lj  ];
            val = center
                + rx * (left  - 2.0f*center + right)
                + ry * (up    - 2.0f*center + down);
        }
        u_new[gi*nx+gj] = val;
    }
}

/* ----------------------------------------------------------------
 * KERNEL 3 — float4 Vectorised Loads
 * 128-bit loads: reads 4 floats per instruction.
 * Each thread processes 4 consecutive columns.
 * Requirement: nx % 4 == 0
 * ---------------------------------------------------------------- */
__global__ void diffuse_float4(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int nx, int ny, float rx, float ry)
{
    const int j4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int i  =  blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j4 + 3 >= nx) return;

    if (i == 0 || i == ny-1) {
        u_new[i*nx+j4  ] = 0.0f; u_new[i*nx+j4+1] = 0.0f;
        u_new[i*nx+j4+2] = 0.0f; u_new[i*nx+j4+3] = 0.0f;
        return;
    }

    /* 128-bit loads */
    const float4 row  = reinterpret_cast<const float4*>(u)[ i   *(nx/4) + j4/4];
    const float4 rowU = reinterpret_cast<const float4*>(u)[(i-1)*(nx/4) + j4/4];
    const float4 rowD = reinterpret_cast<const float4*>(u)[(i+1)*(nx/4) + j4/4];

    const float left0  = (j4 > 0)    ? u[i*nx + j4 - 1] : 0.0f;
    const float right3 = (j4+4 < nx) ? u[i*nx + j4 + 4] : 0.0f;

    const float vals[4]  = { row.x,  row.y,  row.z,  row.w  };
    const float ups[4]   = { rowU.x, rowU.y, rowU.z, rowU.w };
    const float dns[4]   = { rowD.x, rowD.y, rowD.z, rowD.w };
    const float lefts[4] = { left0,  row.x,  row.y,  row.z  };
    const float rights[4]= { row.y,  row.z,  row.w,  right3 };

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int col = j4 + k;
        u_new[i*nx+col] = (col == 0 || col == nx-1) ? 0.0f :
            vals[k]
            + rx * (lefts[k]  - 2.0f*vals[k] + rights[k])
            + ry * (ups[k]    - 2.0f*vals[k] + dns[k]);
    }
}

/* ----------------------------------------------------------------
 * KERNEL 4 — Occupancy Tuned (same fixes as kernel 2 + launch_bounds)
 * Padded SMEM, cooperative load, unified write path, plus
 * __launch_bounds__(256, 4) to cap register use at 64/thread.
 * ---------------------------------------------------------------- */
__global__
__launch_bounds__(256, 4)
void diffuse_occupancy(
    const float * __restrict__ u,
          float * __restrict__ u_new,
    int nx, int ny, float rx, float ry)
{
    __shared__ float s[SMEM][SMEM_PAD];

    const int gi = blockIdx.y * TILE + threadIdx.y;
    const int gj = blockIdx.x * TILE + threadIdx.x;
    const int li = threadIdx.y + 1;
    const int lj = threadIdx.x + 1;

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

    if (gi < ny && gj < nx) {
        float val = 0.0f;
        if (gi > 0 && gi < ny-1 && gj > 0 && gj < nx-1) {
            const float center = s[li  ][lj  ];
            const float left   = s[li  ][lj-1];
            const float right  = s[li  ][lj+1];
            const float up     = s[li-1][lj  ];
            const float down   = s[li+1][lj  ];
            val = center
                + rx * (left  - 2.0f*center + right)
                + ry * (up    - 2.0f*center + down);
        }
        u_new[gi*nx+gj] = val;
    }
}

"""

# Write .cu file to disk
with open("kernels/diffusion_kernels.cu", "w") as f:
    f.write(cu_source)
print("Written: kernels/diffusion_kernels.cu")

# ═══════════════════════════════════════════════════════════════════
# CELL 3 — Compile .cu with nvcc → shared library (.so)
#
# -O3              : full optimisation
# --use_fast_math  : faster transcendentals (safe for diffusion)
# -Xcompiler -fPIC : position-independent code for shared lib
# -shared          : output is a .so loadable by PyCUDA
# ═══════════════════════════════════════════════════════════════════
compile_cmd = [
    "nvcc",
    "-O3",
    "--use_fast_math",
    "-Xcompiler", "-fPIC",
    "-shared",
    "-o", "kernels/diffusion_kernels.so",
    "kernels/diffusion_kernels.cu"
]

print("Compiling with nvcc...")
print("  " + " ".join(compile_cmd))

result = subprocess.run(compile_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("nvcc STDERR:\n", result.stderr)
    raise RuntimeError("nvcc compilation failed — see error above.")

print(f"Compiled successfully → kernels/diffusion_kernels.so")
print(f"Size: {os.path.getsize('kernels/diffusion_kernels.so') / 1024:.1f} KB")

# ═══════════════════════════════════════════════════════════════════
# CELL 4 — Load compiled kernels with PyCUDA
# ═══════════════════════════════════════════════════════════════════
import pycuda.autoinit
import pycuda.driver as drv
import ctypes
import numpy as np

# Load the compiled shared object
lib = ctypes.CDLL("kernels/diffusion_kernels.so")
print(f"Loaded: kernels/diffusion_kernels.so")

# Use PyCUDA's module loading to get kernel function handles
# We reload via SourceModule pointing at the already-compiled PTX
# (cleaner approach: read the .cu and let PyCUDA invoke nvcc)
from pycuda.compiler import SourceModule

with open("kernels/diffusion_kernels.cu", "r") as f:
    cu_code = f.read()

# PyCUDA compiles it internally — this uses the same source
# but gives us clean function handles
mod          = SourceModule(cu_code, options=["-O3", "--use_fast_math"])
k_naive      = mod.get_function("diffuse_naive")
k_shared     = mod.get_function("diffuse_shared")
k_float4     = mod.get_function("diffuse_float4")
k_occupancy  = mod.get_function("diffuse_occupancy")

print("Kernel handles acquired:")
print("  diffuse_naive     ✓")
print("  diffuse_shared    ✓")
print("  diffuse_float4    ✓")
print("  diffuse_occupancy ✓")

# ═══════════════════════════════════════════════════════════════════
# CELL 5 — Helpers
# ═══════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import csv

def gpu_time_ms(func, *args, grid, block, warmup=10, iters=100):
    """
    Accurate GPU kernel timing using CUDA events.
    - warmup iters: prime caches, avoid cold-start bias
    - timed iters:  average to reduce variance
    """
    for _ in range(warmup):
        func(*args, grid=grid, block=block)
    drv.Context.synchronize()

    start = drv.Event()
    stop  = drv.Event()
    start.record()
    for _ in range(iters):
        func(*args, grid=grid, block=block)
    stop.record()
    stop.synchronize()
    return start.time_till(stop) / iters   # ms per call


def gaussian_init(nx, ny, sigma2=0.02):
    """
    Gaussian pulse centred at (0.5, 0.5).
    Dirichlet BC (u=0) enforced on all four edges.
    """
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    u = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / sigma2).astype(np.float32)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return u


def run_solver(kernel, nx, ny, steps, rx, ry,
               block=(16, 16, 1), use_float4=False):
    """
    Allocates GPU buffers, runs `steps` time steps with ping-pong
    buffering, returns (result_field, ms_per_step, GB/s, GFLOP/s).
    """
    u0 = gaussian_init(nx, ny)
    N  = nx * ny

    d0 = drv.mem_alloc(u0.nbytes)
    d1 = drv.mem_alloc(u0.nbytes)
    drv.memcpy_htod(d0, u0)
    drv.memset_d32(d1, 0, N)

    GX = ((nx // 4) + block[0] - 1) // block[0] if use_float4 \
         else (nx + block[0] - 1) // block[0]
    GY = (ny + block[1] - 1) // block[1]

    def step(src, dst):
        kernel(src, dst,
               np.int32(nx), np.int32(ny),
               np.float32(rx), np.float32(ry),
               grid=(GX, GY), block=block)

    # Warmup (not timed)
    for _ in range(10):
        step(d0, d1); d0, d1 = d1, d0
    drv.Context.synchronize()

    # Timed run
    start = drv.Event()
    stop  = drv.Event()
    start.record()
    for _ in range(steps):
        step(d0, d1); d0, d1 = d1, d0
    stop.record()
    stop.synchronize()

    ms_step  = start.time_till(stop) / steps

    # Performance metrics
    # Memory: 5 reads + 1 write = 6 floats = 24 bytes per cell
    bw_gbs   = (6 * N * 4) / (ms_step * 1e-3) / 1e9
    # Compute: 8 FLOPs per interior cell
    gflops   = (8 * N)     / (ms_step * 1e-3) / 1e9

    result = np.empty_like(u0)
    drv.memcpy_dtoh(result, d0)
    d0.free()
    d1.free()

    return result, ms_step, bw_gbs, gflops

print("Helpers defined.")

# ═══════════════════════════════════════════════════════════════════
# CELL 6 — Correctness Verification
# All kernels must match naive within floating-point tolerance.
# ═══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  CORRECTNESS VERIFICATION")
print("=" * 55)

NX_C, NY_C = 256, 256
STEPS_C    = 200
RX_C = RY_C = np.float32(0.24)

r_naive, _, _, _ = run_solver(k_naive,     NX_C, NY_C, STEPS_C, RX_C, RY_C)
r_shared,_, _, _ = run_solver(k_shared,    NX_C, NY_C, STEPS_C, RX_C, RY_C)
r_f4,    _, _, _ = run_solver(k_float4,    NX_C, NY_C, STEPS_C, RX_C, RY_C,
                               use_float4=True)
r_occ,   _, _, _ = run_solver(k_occupancy, NX_C, NY_C, STEPS_C, RX_C, RY_C)

TOL = 1e-4
checks = [
    ("Naive vs Shared",     r_naive, r_shared),
    ("Naive vs Float4",     r_naive, r_f4),
    ("Naive vs Occupancy",  r_naive, r_occ),
]

all_ok = True
for label, a, b in checks:
    err = np.max(np.abs(a - b))
    ok  = err < TOL
    all_ok &= ok
    print(f"  {label:<24} max_diff = {err:.2e}  {'✓' if ok else '✗ FAIL'}")

print()
print("  ✓ All kernels match." if all_ok else "  ✗ Mismatch — review kernel code.")

# ═══════════════════════════════════════════════════════════════════
# CELL 7 — Full Benchmark
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BENCHMARK — 4 Kernels × 5 Grid Sizes")
print("=" * 60)

GRID_SIZES = [128, 256, 512, 1024, 2048]
STEPS_BM   = 200
RX = RY    = np.float32(0.24)

KERNELS = [
    ("Naive",     k_naive,     (16,16,1), False),
    ("Shared",    k_shared,    (16,16,1), False),
    ("Float4",    k_float4,    (16,16,1), True),
    ("Occupancy", k_occupancy, (16,16,1), False),
]

results = {k[0]: {} for k in KERNELS}

print(f"  {'Kernel':12}  {'Grid':>10}  {'ms/step':>9}  "
      f"{'GB/s':>8}  {'GFLOP/s':>9}  {'Speedup':>9}")
print("  " + "-" * 65)

for kname, kfunc, kblock, kf4 in KERNELS:
    for nx in GRID_SIZES:
        if kf4 and nx % 4 != 0:
            continue
        ny = nx
        _, ms, bw, gf = run_solver(kfunc, nx, ny, STEPS_BM,
                                   RX, RY, block=kblock,
                                   use_float4=kf4)
        results[kname][nx] = (ms, bw, gf)

        base_ms  = results["Naive"].get(nx, (ms,))[0]
        speedup  = base_ms / ms
        sp_str   = "baseline" if kname == "Naive" else f"{speedup:.2f}x"

        print(f"  {kname:12}  {nx:4d}x{nx:<4d}  "
              f"{ms:9.3f}  {bw:8.1f}  {gf:9.2f}  {sp_str:>9}")
    print()

# ═══════════════════════════════════════════════════════════════════
# CELL 8 — Save CSV
# ═══════════════════════════════════════════════════════════════════
with open("benchmarks/optimization_stages.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel","nx","ny","steps",
                "ms_per_step","GB_per_s","GFLOP_per_s","speedup_vs_naive"])
    for kname, _, _, _ in KERNELS:
        for nx in GRID_SIZES:
            if nx not in results[kname]: continue
            ms, bw, gf = results[kname][nx]
            base = results["Naive"].get(nx, (ms,))[0]
            w.writerow([kname, nx, nx, STEPS_BM,
                        f"{ms:.4f}", f"{bw:.2f}",
                        f"{gf:.2f}", f"{base/ms:.3f}"])

print("Saved: benchmarks/optimization_stages.csv")

# ═══════════════════════════════════════════════════════════════════
# CELL 9 — Performance Charts
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "Naive"    : "#888888",
    "Shared"   : "#2196F3",
    "Float4"   : "#4CAF50",
    "Occupancy": "#FF9800",
}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Chart 1: ms/step ─────────────────────────────────────────────
ax = axes[0]
for kname, _, _, _ in KERNELS:
    xs = sorted(results[kname].keys())
    ys = [results[kname][x][0] for x in xs]
    ax.plot(xs, ys, marker='o', label=kname,
            color=COLORS[kname], linewidth=2, markersize=7)
ax.set_title("Execution Time per Step", fontsize=12)
ax.set_xlabel("Grid size")
ax.set_ylabel("ms / step")
ax.set_xticks(GRID_SIZES)
ax.set_xticklabels([f"{n}²" for n in GRID_SIZES], rotation=30)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Chart 2: Speedup over naive ───────────────────────────────────
ax = axes[1]
for kname, _, _, _ in KERNELS:
    if kname == "Naive": continue
    xs    = sorted(results[kname].keys())
    bases = [results["Naive"].get(x, (None,))[0] for x in xs]
    ys    = [b / results[kname][x][0]
             for x, b in zip(xs, bases) if b is not None]
    xs    = [x for x, b in zip(xs, bases) if b is not None]
    ax.plot(xs, ys, marker='s', label=f"{kname}",
            color=COLORS[kname], linewidth=2, markersize=7)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Naive (1×)')
ax.set_title("Speedup over Naive Kernel", fontsize=12)
ax.set_xlabel("Grid size")
ax.set_ylabel("Speedup (×)")
ax.set_xticks(GRID_SIZES)
ax.set_xticklabels([f"{n}²" for n in GRID_SIZES], rotation=30)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Chart 3: Achieved bandwidth ───────────────────────────────────
ax = axes[2]
for kname, _, _, _ in KERNELS:
    xs = sorted(results[kname].keys())
    ys = [results[kname][x][1] for x in xs]
    ax.plot(xs, ys, marker='^', label=kname,
            color=COLORS[kname], linewidth=2, markersize=7)
ax.axhline(265, color='red', linestyle='--',
           linewidth=1.5, label='T4 peak (~265 GB/s)')
ax.set_title("Achieved Memory Bandwidth", fontsize=12)
ax.set_xlabel("Grid size")
ax.set_ylabel("GB/s")
ax.set_xticks(GRID_SIZES)
ax.set_xticklabels([f"{n}²" for n in GRID_SIZES], rotation=30)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(
    "2D Heat Diffusion — CUDA Optimization Stages (T4 GPU)\n"
    "Source: kernels/diffusion_kernels.cu  |  Compiled: nvcc -O3 --use_fast_math",
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("benchmarks/performance.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: benchmarks/performance.png")

# ═══════════════════════════════════════════════════════════════════
# CELL 10 — Diffusion Visualisation
# ═══════════════════════════════════════════════════════════════════
print("Running visualisation simulation (512×512, 2000 steps)...")

NX_V   = 512
ALPHA  = 0.01
DX     = 1.0 / NX_V
DT     = 0.0001
RX_V   = np.float32(ALPHA * DT / DX**2)
RY_V   = RX_V

SNAP_AT = [0, 100, 500, 2000]
snaps   = {}

u0 = gaussian_init(NX_V, NX_V)
d0 = drv.mem_alloc(u0.nbytes)
d1 = drv.mem_alloc(u0.nbytes)
drv.memcpy_htod(d0, u0)
drv.memset_d32(d1, 0, NX_V * NX_V)
snaps[0] = u0.copy()

BX, BY = 16, 16
GX = (NX_V + BX - 1) // BX
GY = (NX_V + BY - 1) // BY

done = 0
for target in SNAP_AT[1:]:
    for _ in range(target - done):
        k_shared(d0, d1,
                 np.int32(NX_V), np.int32(NX_V), RX_V, RY_V,
                 grid=(GX, GY), block=(BX, BY, 1))
        d0, d1 = d1, d0
    done = target
    snap = np.empty(NX_V * NX_V, dtype=np.float32)
    drv.memcpy_dtoh(snap, d0)
    snaps[target] = snap.reshape(NX_V, NX_V)

d0.free(); d1.free()

cmap = LinearSegmentedColormap.from_list(
    'heat', ['#000033','#003399','#00ccff','#ffff00','#ff4400','#ffffff'])

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (step, field) in zip(axes, snaps.items()):
    im = ax.imshow(field, cmap=cmap, origin='lower',
                   vmin=0, vmax=1, interpolation='bilinear')
    ax.set_title(f"t = {step * DT:.4f} s\nstep {step}", fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f"2D Heat Diffusion — Gaussian Pulse Spreading\n"
    f"Grid: {NX_V}×{NX_V}  |  α={ALPHA}  |  Kernel: diffuse_shared",
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("benchmarks/diffusion_snapshots.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: benchmarks/diffusion_snapshots.png")

# ═══════════════════════════════════════════════════════════════════
# CELL 11 — Summary
# ═══════════════════════════════════════════════════════════════════
NX_S = 1024
print("\n" + "=" * 62)
print(f"  OPTIMIZATION SUMMARY  —  {NX_S}×{NX_S} grid")
print("=" * 62)
print(f"  {'Stage':<12}  {'What it does':<32}  {'ms':>6}  {'Speedup':>8}")
print("  " + "-" * 60)

desc = {
    "Naive"    : "Global memory, no reuse",
    "Shared"   : "Shared mem tile + halo load",
    "Float4"   : "128-bit vectorised loads",
    "Occupancy": "__launch_bounds__ hint",
}
base = results["Naive"].get(NX_S, (1,))[0]
for kname, _, _, _ in KERNELS:
    if NX_S not in results[kname]: continue
    ms  = results[kname][NX_S][0]
    sp  = base / ms
    tag = " ← baseline" if kname == "Naive" else ""
    print(f"  {kname:<12}  {desc[kname]:<32}  {ms:6.3f}  {sp:7.2f}×{tag}")

print()
print("  Output files:")
print("  • benchmarks/optimization_stages.csv")
print("  • benchmarks/performance.png")
print("  • benchmarks/diffusion_snapshots.png")
print("  • kernels/diffusion_kernels.cu   ← the pure CUDA source")
print("=" * 62)
print("\n  ✓ Complete. Ready to build Navier-Stokes on top of this.")
