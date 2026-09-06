// full DNA pose scoring kernel (near pairs + far field) for LKlight-CUDA.
// One thread per *ligand atom*:
//   - near (d <= 10 A): scans the receptor cell-list neighbourhood (27 cells of
//     10 A) and accumulates the exact per-pair terms with the SAME formulas as
//     the CPU grid path (clamped electrostatics, LJ capped at 1.0, heavy-atom
//     linear clash penalty).
//   - far (10 < d <= 30 A): trilinear gather of the receptor field grid phi.
// out[j] = elec_raw_j (near clamped sum + far q*phi) and vdw_j are atom-local;
// the host reduces them. Interface flags / restraints / membrane are handled
// by the host (CPU grid path) when needed.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>   // uintptr_t (persistent buffer-cache keys)

#define NEAR2  100.0f        // 10 A squared (near cutoff)
#define LJ_CAP 1.0f
#define CP_W   6.0f
#define CP_F   0.75f
#define ES_CAP 0.01204819f   // 1.0*EPSILON/FACTOR with EPSILON=4, FACTOR=332

// Per-family kernel flags (the LJ term is always on: every all-atom family
// scores it). DNA = FAR|ELEC|CLASH (7), PYDOCK = FAR|ELEC (3), VDW = 0 (LJ only).
#define F_FAR   1u
#define F_ELEC  2u
#define F_CLASH 4u

__device__ __forceinline__ float sample_phi_f(
    const float* __restrict__ phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    float x, float y, float z)
{
    float fx = (x - ox) / sp;
    float fy = (y - oy) / sp;
    float fz = (z - oz) / sp;
    if (fx < 0.f || fy < 0.f || fz < 0.f) return 0.f;
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    if (ix + 1 >= nx || iy + 1 >= ny || iz + 1 >= nz) return 0.f;
    float tx = fx - ix, ty = fy - iy, tz = fz - iz;
    size_t base = (size_t)iz * ny * nx + (size_t)iy * nx + ix;
    float c000 = phi[base], c100 = phi[base + 1];
    float c010 = phi[base + nx], c110 = phi[base + nx + 1];
    float c001 = phi[base + (size_t)ny * nx], c101 = phi[base + (size_t)ny * nx + 1];
    float c011 = phi[base + (size_t)ny * nx + nx], c111 = phi[base + (size_t)ny * nx + nx + 1];
    float c00 = c000 * (1.f - tx) + c100 * tx;
    float c10 = c010 * (1.f - tx) + c110 * tx;
    float c01 = c001 * (1.f - tx) + c101 * tx;
    float c11 = c011 * (1.f - tx) + c111 * tx;
    float c0 = c00 * (1.f - ty) + c10 * ty;
    float c1 = c01 * (1.f - ty) + c11 * ty;
    return c0 * (1.f - tz) + c1 * tz;
}

__global__ void full_score_kernel(
    const float* __restrict__ phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* __restrict__ r_coords,   // nr*3
    const float* __restrict__ r_ele,      // nr
    const float* __restrict__ r_svdw,     // nr
    const float* __restrict__ r_vdwr,     // nr
    const unsigned char* __restrict__ r_heavy,  // nr
    const int* __restrict__ cell_start,   // (ncx+1)*(ncy+1)*(ncz+1) prefix
    const int* __restrict__ cell_atoms,   // flat atom ids
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,   // cell grid geometry (10 A)
    const float* __restrict__ l_coords,   // nl*3 (already transformed)
    const float* __restrict__ l_ele,      // nl
    const float* __restrict__ l_svdw,     // nl
    const float* __restrict__ l_vdwr,     // nl
    const unsigned char* __restrict__ l_heavy,  // nl
    int nl,
    float* __restrict__ out_elec,         // nl atom-local raw electrostatics
    float* __restrict__ out_vdw)          // nl atom-local vdw incl. clash
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nl) return;
    float x = l_coords[j*3], y = l_coords[j*3+1], z = l_coords[j*3+2];
    float qj = l_ele[j];
    float svdwj = l_svdw[j];
    float vdwrj = l_vdwr[j];
    bool hj = l_heavy[j] != 0;

    float elec = 0.f, vdw = 0.f;

    // far field (host field grid): q_j * phi(x_j)
    if (qj != 0.f)
        elec += qj * sample_phi_f(phi, nx, ny, nz, ox, oy, oz, sp, x, y, z);

    // near: receptor cell list around (x,y,z)
    int cxi = (int)floorf((x - c_ox) / c_sp);
    int cyi = (int)floorf((y - c_oy) / c_sp);
    int czi = (int)floorf((z - c_oz) / c_sp);
    for (int dz = -1; dz <= 1; ++dz) {
        int czz = czi + dz;
        if (czz < 0 || czz >= ncz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int cyy = cyi + dy;
            if (cyy < 0 || cyy >= ncy) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int cxx = cxi + dx;
                if (cxx < 0 || cxx >= ncx) continue;
                int cell = (czz * ncy + cyy) * ncx + cxx;
                int b = cell_start[cell];
                int e = cell_start[cell + 1];
                for (int p = b; p < e; ++p) {
                    int i = cell_atoms[p];
                    float dxf = x - r_coords[i*3];
                    float dyf = y - r_coords[i*3+1];
                    float dzf = z - r_coords[i*3+2];
                    float d2 = dxf*dxf + dyf*dyf + dzf*dzf;
                    if (d2 <= NEAR2) {
                        // clamped electrostatics (same as CPU)
                        float ae = qj * r_ele[i] / d2;
                        if (ae > ES_CAP) ae = ES_CAP;
                        else if (ae < -ES_CAP) ae = -ES_CAP;
                        elec += ae;
                        // LJ with 1.0 cap (same as CPU: p6 = (vdw_r/d)^6)
                        float sv = svdwj * r_svdw[i];
                        float rr = vdwrj + r_vdwr[i];
                        float rr2 = rr * rr;
                        float p6 = rr2 * rr2 * rr2 / (d2 * d2 * d2);
                        float p6sq = p6*p6;
                        float vp = sv * (p6sq - 2.0f * p6);
                        if (vp > LJ_CAP) vp = LJ_CAP;
                        // heavy-atom linear clash penalty
                        if (hj && r_heavy[i]) {
                            float d = sqrtf(d2);
                            float dmin = CP_F * rr;
                            if (d < dmin) vp += CP_W * (dmin - d);
                        }
                        vdw += vp;
                    }
                }
            }
        }
    }
    out_elec[j] = elec;
    out_vdw[j] = vdw;
}

extern "C" int cuda_full_score(
    const float* phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* r_coords, const float* r_ele, const float* r_svdw,
    const float* r_vdwr, const unsigned char* r_heavy, int nr,
    const int* cell_start, const int* cell_atoms,
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,
    const float* l_coords, const float* l_ele, const float* l_svdw,
    const float* l_vdwr, const unsigned char* l_heavy, int nl,
    float* out_elec, float* out_vdw)
{
    cudaError_t err;
    float *d_phi=0,*d_rc=0,*d_re=0,*d_rsv=0,*d_rv=0; unsigned char *d_rh=0;
    int *d_cs=0,*d_ca=0;
    float *d_lc=0,*d_le=0,*d_lsv=0,*d_lv=0; unsigned char *d_lh=0;
    float *d_oe=0,*d_ov=0;
    size_t phi_b=(size_t)nx*ny*nz*sizeof(float);
    size_t cb=(size_t)((ncx+1)*(ncy+1)*(ncz+1))*sizeof(int);

#define CK(expr) do { err=(expr); if(err!=cudaSuccess) goto fail; } while(0)
    CK(cudaMalloc(&d_phi,phi_b));
    CK(cudaMalloc(&d_rc,(size_t)nr*3*sizeof(float)));
    CK(cudaMalloc(&d_re,(size_t)nr*sizeof(float)));
    CK(cudaMalloc(&d_rsv,(size_t)nr*sizeof(float)));
    CK(cudaMalloc(&d_rv,(size_t)nr*sizeof(float)));
    CK(cudaMalloc(&d_rh,(size_t)nr));
    CK(cudaMalloc(&d_cs,cb));
    CK(cudaMalloc(&d_ca,(size_t)nr*sizeof(int)));
    CK(cudaMalloc(&d_lc,(size_t)nl*3*sizeof(float)));
    CK(cudaMalloc(&d_le,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lsv,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lv,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lh,(size_t)nl));
    CK(cudaMalloc(&d_oe,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_ov,(size_t)nl*sizeof(float)));
    CK(cudaMemcpy(d_phi,phi,phi_b,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rc,r_coords,(size_t)nr*3*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_re,r_ele,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rsv,r_svdw,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rv,r_vdwr,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rh,r_heavy,(size_t)nr,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cs,cell_start,cb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ca,cell_atoms,(size_t)nr*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lc,l_coords,(size_t)nl*3*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_le,l_ele,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lsv,l_svdw,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv,l_vdwr,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lh,l_heavy,(size_t)nl,cudaMemcpyHostToDevice));
    {
        int threads=256;
        int blocks=(nl+threads-1)/threads;
        full_score_kernel<<<blocks,threads>>>(
            d_phi,nx,ny,nz,ox,oy,oz,sp,
            d_rc,d_re,d_rsv,d_rv,d_rh,
            d_cs,d_ca,ncx,ncy,ncz,c_ox,c_oy,c_oz,c_sp,
            d_lc,d_le,d_lsv,d_lv,d_lh,nl,
            d_oe,d_ov);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpy(out_elec,d_oe,(size_t)nl*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out_vdw,d_ov,(size_t)nl*sizeof(float),cudaMemcpyDeviceToHost));
    cudaFree(d_phi);cudaFree(d_rc);cudaFree(d_re);cudaFree(d_rsv);cudaFree(d_rv);cudaFree(d_rh);
    cudaFree(d_cs);cudaFree(d_ca);cudaFree(d_lc);cudaFree(d_le);cudaFree(d_lsv);cudaFree(d_lv);cudaFree(d_lh);
    cudaFree(d_oe);cudaFree(d_ov);
    return 0;
fail:
    { const char* m=cudaGetErrorString(err); fprintf(stderr,"cuda_full_score error: %s\n",m); }
    if(d_phi)cudaFree(d_phi);if(d_rc)cudaFree(d_rc);if(d_re)cudaFree(d_re);if(d_rsv)cudaFree(d_rsv);
    if(d_rv)cudaFree(d_rv);if(d_rh)cudaFree(d_rh);if(d_cs)cudaFree(d_cs);if(d_ca)cudaFree(d_ca);
    if(d_lc)cudaFree(d_lc);if(d_le)cudaFree(d_le);if(d_lsv)cudaFree(d_lsv);if(d_lv)cudaFree(d_lv);
    if(d_lh)cudaFree(d_lh);if(d_oe)cudaFree(d_oe);if(d_ov)cudaFree(d_ov);
    return -1;
#undef CK
}

// ── batched variant ─────────────────────────────────────────────────────────
// gridDim = (blocks over ligand atoms, pose index). One kernel launch scores N
// poses: amortises launch + host-device traffic; this is what makes the GPU win
// (per-pose launches pay ~18 ms sync each and lose to the CPU grid path).
__global__ void batch_full_score_kernel(
    const float* __restrict__ phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* __restrict__ r_coords, const float* __restrict__ r_ele,
    const float* __restrict__ r_svdw, const float* __restrict__ r_vdwr,
    const unsigned char* __restrict__ r_heavy,
    const int* __restrict__ cell_start, const int* __restrict__ cell_atoms,
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,
    const float* __restrict__ l_base,          // nl * 3 (reference ligand coords)
    const double* __restrict__ poses,          // N * 7  (w,x,y,z,tx,ty,tz)
    const float* __restrict__ l_ele, const float* __restrict__ l_svdw,
    const float* __restrict__ l_vdwr, const unsigned char* __restrict__ l_heavy,
    int nl, int N,
    unsigned flags,
    double* __restrict__ out)                 // N * 2  (elec, vdw)
{
    int pose = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // OPT (2026-09-05): per-block shared-memory reduction replaces one device
    // atomicAdd per *ligand atom* with one atomicAdd per *block* into the pose
    // accumulator. With gridDim.y = N poses x ~nl/256 blocks, this cuts the
    // number of serialized atomic ops on each pose's out[pose*2] slot from ~nl
    // to ~ceil(nl/256) — a large win at high batch throughput (N=1000). Out-of
    // -range threads (partial last block) contribute 0 and still join the
    // __syncthreads barrier. The per-atom physics (elec/vdw accumulators) is
    // byte-for-byte identical to the single-pose full_score_kernel; only the
    // intra-block summation order of the f32 atom locals changes (block-tree
    // order instead of per-atom atomics), which stays within the same f32 round
    // -off budget already accepted between GPU and CPU grid paths.
    extern __shared__ float sm[];   // 2*blockDim.x floats: elec then vdw
    float* s_elec = sm;
    float* s_vdw  = sm + blockDim.x;
    if (j >= nl) { s_elec[threadIdx.x] = 0.f; s_vdw[threadIdx.x] = 0.f; }
    else {
    // Rigid transform on the device: v' = R(q)·v + t, in double precision so the
    // result matches the host-side f64 transform bit-for-bit before the f32 cast
    // (avoids uploading N*nl*3 transformed coords every step ~ 30 MB @ n=200).
    const double* pq = poses + (size_t)pose * 7;
    double w = pq[0], qx = pq[1], qy = pq[2], qz = pq[3];
    double tx = pq[4], ty = pq[5], tz = pq[6];
    const float* lb = l_base + (size_t)j * 3;
    double vx = lb[0], vy = lb[1], vz = lb[2];
    double m00 = 1. - 2.*(qy*qy+qz*qz), m01 = 2.*(qx*qy - w*qz),  m02 = 2.*(qx*qz + w*qy);
    double m10 = 2.*(qx*qy + w*qz),      m11 = 1. - 2.*(qx*qx+qz*qz), m12 = 2.*(qy*qz - w*qx);
    double m20 = 2.*(qx*qz - w*qy),      m21 = 2.*(qy*qz + w*qx),  m22 = 1. - 2.*(qx*qx+qy*qy);
    float x = (float)(m00*vx + m01*vy + m02*vz + tx);
    float y = (float)(m10*vx + m11*vy + m12*vz + ty);
    float z = (float)(m20*vx + m21*vy + m22*vz + tz);
    float qj = l_ele[j];
    float svdwj = l_svdw[j];
    float vdwrj = l_vdwr[j];
    bool hj = l_heavy[j] != 0;
    float elec = 0.f, vdw = 0.f;
    if ((flags & F_FAR) && qj != 0.f)
        elec += qj * sample_phi_f(phi, nx, ny, nz, ox, oy, oz, sp, x, y, z);
    int cxi = (int)floorf((x - c_ox) / c_sp);
    int cyi = (int)floorf((y - c_oy) / c_sp);
    int czi = (int)floorf((z - c_oz) / c_sp);
    for (int dz = -1; dz <= 1; ++dz) {
        int czz = czi + dz;
        if (czz < 0 || czz >= ncz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int cyy = cyi + dy;
            if (cyy < 0 || cyy >= ncy) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int cxx = cxi + dx;
                if (cxx < 0 || cxx >= ncx) continue;
                int cell = (czz * ncy + cyy) * ncx + cxx;
                int b = cell_start[cell], e = cell_start[cell + 1];
                for (int p = b; p < e; ++p) {
                    int i = cell_atoms[p];
                    float dxf = x - r_coords[i*3];
                    float dyf = y - r_coords[i*3+1];
                    float dzf = z - r_coords[i*3+2];
                    float d2 = dxf*dxf + dyf*dyf + dzf*dzf;
                    if (d2 <= NEAR2) {
                        if (flags & F_ELEC) {
                            float ae = qj * r_ele[i] / d2;
                            if (ae > ES_CAP) ae = ES_CAP;
                            else if (ae < -ES_CAP) ae = -ES_CAP;
                            elec += ae;
                        }
                        float sv = svdwj * r_svdw[i];
                        float rr = vdwrj + r_vdwr[i];
                        float rr2 = rr * rr;
                        float p6 = rr2 * rr2 * rr2 / (d2 * d2 * d2);
                        float vp = sv * (p6*p6 - 2.0f*p6);
                        if (vp > LJ_CAP) vp = LJ_CAP;
                        if ((flags & F_CLASH) && hj && r_heavy[i]) {
                            float d = sqrtf(d2);
                            float dmin = CP_F * rr;
                            if (d < dmin) vp += CP_W * (dmin - d);
                        }
                        vdw += vp;
                    }
                }
            }
        }
    }
        s_elec[threadIdx.x] = elec;
        s_vdw[threadIdx.x]  = vdw;
    } // end per-atom branch (out-of-range threads already stored 0 above)
    __syncthreads();
    // Block-tree reduction over elec then vdw (stride halving).
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_elec[threadIdx.x] += s_elec[threadIdx.x + s];
            s_vdw[threadIdx.x]  += s_vdw[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&out[(size_t)pose * 2], (double)s_elec[0]);
        atomicAdd(&out[(size_t)pose * 2 + 1], (double)s_vdw[0]);
    }
}

// ── persistent device-buffer cache ──────────────────────────────────────────
// OPT (2026-09-05): over a GSO run, `cuda_batch_score` is called once per step
// with the SAME receptor far-field grid + receptor cell list + ligand reference
// (the host holds them in stable OnceLock buffers). Previously every call
// cudaMalloc'd and cudaMemcpy'd all of phi/receptor/cells/ligand base from host
// to device (~tens-to-hundreds of MB at 0.5 A for large receptors) and freed
// them after — repeated ~once per step × 1000 steps. Now the constant device
// arrays are cached and only (re)uploaded when the host buffers' identity
// changes (the ptr key below). Per step we upload only the small N×7 pose
// double buffer and download the N×2 out buffer, cutting per-step PCIe traffic
// to a tiny fraction. Process exits free device memory via cudaDeviceReset (OS
// reclaim) — acceptable for the CLI/docking-process model.
typedef struct { int live; float* phi; float* rc; float* re; float* rsv;
                 float* rv; unsigned char* rh; int* cs; int* ca;
                 float* lb; float* le; float* lsv; float* lv; unsigned char* lh;
                 double* out;
                 unsigned long long key_phi, key_rc, key_cs, key_lb, key_nl;
                 unsigned flags;
                 int n_phi, n_rc, n_cs, n_lb, n_l; } BatchCache;
static BatchCache g_bc = {0};

extern "C" int cuda_batch_score(
    const float* phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* r_coords, const float* r_ele, const float* r_svdw,
    const float* r_vdwr, const unsigned char* r_heavy, int nr,
    const int* cell_start, const int* cell_atoms,
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,
    const float* l_base, const double* poses, const float* l_ele,
    const float* l_svdw, const float* l_vdwr, const unsigned char* l_heavy,
    int nl, int N,
    unsigned flags,
    double* out)
{
    cudaError_t err;
    int stage = 0;   // 0=alloc-fresh 1=malloc-poses 2=copy-poses 3=memset-out 4=launch 5=sync 6=readback
    cudaGetLastError();  // drain any stale sticky error so each step is independent
    double *d_ps=0;
    size_t phi_b=(size_t)nx*ny*nz*sizeof(float);
    size_t cb=(size_t)((ncx+1)*(ncy+1)*(ncz+1))*sizeof(int);
    // host-buffer identity key for the constant (receptor + ligand-base) arrays
    unsigned long long k_phi = (unsigned long long)(uintptr_t)phi;
    unsigned long long k_rc  = (unsigned long long)(uintptr_t)r_coords;
    unsigned long long k_cs  = (unsigned long long)(uintptr_t)cell_start;
    unsigned long long k_lb  = (unsigned long long)(uintptr_t)l_base;
    int nl_k = nl;
    int fresh = !(g_bc.live &&
                  g_bc.key_phi==k_phi && g_bc.key_rc==k_rc && g_bc.key_cs==k_cs &&
                  g_bc.key_lb==k_lb && g_bc.n_l==nl_k && g_bc.flags==flags);

    if (fresh) {
        // release any previous receptor's cached buffers
        if (g_bc.live) {
            cudaFree(g_bc.phi);cudaFree(g_bc.rc);cudaFree(g_bc.re);cudaFree(g_bc.rsv);
            cudaFree(g_bc.rv);cudaFree(g_bc.rh);cudaFree(g_bc.cs);cudaFree(g_bc.ca);
            cudaFree(g_bc.lb);cudaFree(g_bc.le);cudaFree(g_bc.lsv);cudaFree(g_bc.lv);
            cudaFree(g_bc.lh);cudaFree(g_bc.out);
            memset(&g_bc,0,sizeof(g_bc));
        }
#define CK(expr) do { err=(expr); if(err!=cudaSuccess) goto fail; } while(0)
        CK(cudaMalloc(&g_bc.phi,phi_b));
        CK(cudaMalloc(&g_bc.rc,(size_t)nr*3*sizeof(float)));
        CK(cudaMalloc(&g_bc.re,(size_t)nr*sizeof(float)));
        CK(cudaMalloc(&g_bc.rsv,(size_t)nr*sizeof(float)));
        CK(cudaMalloc(&g_bc.rv,(size_t)nr*sizeof(float)));
        CK(cudaMalloc(&g_bc.rh,(size_t)nr));
        CK(cudaMalloc(&g_bc.cs,cb));
        CK(cudaMalloc(&g_bc.ca,(size_t)nr*sizeof(int)));
        CK(cudaMalloc(&g_bc.lb,(size_t)nl*3*sizeof(float)));
        CK(cudaMalloc(&g_bc.le,(size_t)nl*sizeof(float)));
        CK(cudaMalloc(&g_bc.lsv,(size_t)nl*sizeof(float)));
        CK(cudaMalloc(&g_bc.lv,(size_t)nl*sizeof(float)));
        CK(cudaMalloc(&g_bc.lh,(size_t)nl));
        CK(cudaMalloc(&g_bc.out,(size_t)((nl>N?nl:N))*2*sizeof(double)));
        CK(cudaMemcpy(g_bc.phi,phi,phi_b,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.rc,r_coords,(size_t)nr*3*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.re,r_ele,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.rsv,r_svdw,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.rv,r_vdwr,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.rh,r_heavy,(size_t)nr,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.cs,cell_start,cb,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.ca,cell_atoms,(size_t)nr*sizeof(int),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.lb,l_base,(size_t)nl*3*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.le,l_ele,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.lsv,l_svdw,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.lv,l_vdwr,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_bc.lh,l_heavy,(size_t)nl,cudaMemcpyHostToDevice));
        g_bc.key_phi=k_phi; g_bc.key_rc=k_rc; g_bc.key_cs=k_cs;
        g_bc.key_lb=k_lb;   g_bc.n_l=nl_k;    g_bc.flags=flags; g_bc.live=1;
#undef CK
    }

    // per-step: upload only the small N×7 pose buffer, zero out, launch, readback.
    stage = 1;
    err = cudaMalloc(&d_ps,(size_t)N*7*sizeof(double));
    if (err != cudaSuccess) goto fail;
    stage = 2;
    err = cudaMemcpy(d_ps,poses,(size_t)N*7*sizeof(double),cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_ps); goto fail; }
    stage = 3;
    err = cudaMemset(g_bc.out,0,(size_t)N*2*sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_ps); goto fail; }
    {
        int threads = 256;
        int bx = (nl + threads - 1) / threads;
        dim3 grid(bx, N);
        stage = 4;
        batch_full_score_kernel<<<grid, threads, (size_t)2 * threads * sizeof(float)>>>(
            g_bc.phi,nx,ny,nz,ox,oy,oz,sp,
            g_bc.rc,g_bc.re,g_bc.rsv,g_bc.rv,g_bc.rh,g_bc.cs,g_bc.ca,
            ncx,ncy,ncz,c_ox,c_oy,c_oz,c_sp,
            g_bc.lb,d_ps,g_bc.le,g_bc.lsv,g_bc.lv,g_bc.lh,nl,N,flags,g_bc.out);
        err = cudaGetLastError();          // launch-config errors surface here
        if (err != cudaSuccess) { cudaFree(d_ps); goto fail; }
        stage = 5;
        err = cudaDeviceSynchronize();     // kernel execution errors surface here
        if (err != cudaSuccess) { cudaFree(d_ps); goto fail; }
    }
    stage = 6;
    err = cudaMemcpy(out,g_bc.out,(size_t)N*2*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(d_ps);
    if (err != cudaSuccess) goto fail;
    return 0;
fail:
    { const char* m=cudaGetErrorString(err); fprintf(stderr,"cuda_batch_score error stage=%d N=%d nl=%d err=%d(%s)\n",stage,N,nl,(int)err,m); }
    if (d_ps) cudaFree(d_ps);
    return -1;
}

// ── CPYDOCK desolvation (two-stage) ──────────────────────────────────────────
// Stage A collects, per pose, the minimum squared distance to an opposite-side
// atom over "non-excluded" pairs — CPYDOCK's C-binary-compatible exclusion is
// flag(i) = (i even) && hydrogens[i/2]!=0 — for every receptor atom i
// (device-wide atomic float-min on the integer bits of d2>0, which preserves
// order) and every ligand atom j (thread-private, single writer). Stage B
// reduces the per-pose desolvation S = Σ g(min)·des over receptor + ligand,
// where g(d2) = d2<=SOLV2 && d2>0 && asa>0 ? min(-10·√d2+65, asa) : 0.
// The host combines score = -(E·332/4 + 0.1·V − S).
__global__ void cpydock_min_kernel(
    const float* __restrict__ r_coords, const unsigned char* __restrict__ r_flag,
    const int* __restrict__ cell_start, const int* __restrict__ cell_atoms,
    int ncx, int ncy, int ncz, float c_ox, float c_oy, float c_oz, float c_sp,
    const float* __restrict__ l_base, const double* __restrict__ poses,
    const unsigned char* __restrict__ l_flag, int nr, int nl, int N,
    int* __restrict__ dminR, float* __restrict__ dminL)
{
    int pose = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nl) return;
    const double* pq = poses + (size_t)pose * 7;
    double w = pq[0], qx = pq[1], qy = pq[2], qz = pq[3];
    double tx = pq[4], ty = pq[5], tz = pq[6];
    const float* lb = l_base + (size_t)j * 3;
    double vx = lb[0], vy = lb[1], vz = lb[2];
    double m00 = 1. - 2.*(qy*qy+qz*qz), m01 = 2.*(qx*qy - w*qz),  m02 = 2.*(qx*qz + w*qy);
    double m10 = 2.*(qx*qy + w*qz),      m11 = 1. - 2.*(qx*qx+qz*qz), m12 = 2.*(qy*qz - w*qx);
    double m20 = 2.*(qx*qz - w*qy),      m21 = 2.*(qy*qz + w*qx),  m22 = 1. - 2.*(qx*qx+qy*qy);
    float x = (float)(m00*vx + m01*vy + m02*vz + tx);
    float y = (float)(m10*vx + m11*vy + m12*vz + ty);
    float z = (float)(m20*vx + m21*vy + m22*vz + tz);
    bool lf = l_flag[j] != 0;
    float minL = 3.4e38f;
    int cxi = (int)floorf((x - c_ox) / c_sp);
    int cyi = (int)floorf((y - c_oy) / c_sp);
    int czi = (int)floorf((z - c_oz) / c_sp);
    for (int dz = -1; dz <= 1; ++dz) {
        int czz = czi + dz;
        if (czz < 0 || czz >= ncz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int cyy = cyi + dy;
            if (cyy < 0 || cyy >= ncy) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int cxx = cxi + dx;
                if (cxx < 0 || cxx >= ncx) continue;
                int cell = (czz * ncy + cyy) * ncx + cxx;
                int b = cell_start[cell], e = cell_start[cell + 1];
                for (int p = b; p < e; ++p) {
                    int i = cell_atoms[p];
                    float dxf = x - r_coords[i*3];
                    float dyf = y - r_coords[i*3+1];
                    float dzf = z - r_coords[i*3+2];
                    float d2 = dxf*dxf + dyf*dyf + dzf*dzf;
                    if (!lf && r_flag[i] == 0 && d2 < minL) {
                        minL = d2;   // ligand-atom private minimum
                    }
                    if (!lf && r_flag[i] == 0) {
                        // receptor-atom minimum: int-ordered atomic min (d2>0)
                        atomicMin(&dminR[(size_t)pose * nr + i], __float_as_int(d2));
                    }
                }
            }
        }
    }
    dminL[(size_t)pose * nl + j] = minL;
}

__global__ void cpydock_solv_kernel(
    const int* __restrict__ dminR, const float* __restrict__ dminL,
    const float* __restrict__ r_asa, const float* __restrict__ r_des,
    const float* __restrict__ l_asa, const float* __restrict__ l_des,
    int nr, int nl, int N,
    double* __restrict__ outS)
{
    int pose = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sm[];
    float acc = 0.f;
    for (int i = tid; i < nr; i += gridDim.x * blockDim.x) {
        float d2 = __int_as_float(dminR[(size_t)pose * nr + i]);
        if (d2 <= 40.96f && d2 > 0.f && r_asa[i] > 0.f) {
            float sv = -10.f * sqrtf(d2) + 65.f;
            if (sv > r_asa[i]) sv = r_asa[i];
            acc += sv * r_des[i];
        }
    }
    for (int j = tid; j < nl; j += gridDim.x * blockDim.x) {
        float d2 = dminL[(size_t)pose * nl + j];
        if (d2 <= 40.96f && d2 > 0.f && l_asa[j] > 0.f) {
            float sv = -10.f * sqrtf(d2) + 65.f;
            if (sv > l_asa[j]) sv = l_asa[j];
            acc += sv * l_des[j];
        }
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&outS[pose], (double)sm[0]);
}

extern "C" int cuda_cpydock_solv(
    const float* r_coords, const unsigned char* r_flag, int nr,
    const int* cell_start, const int* cell_atoms,
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,
    const float* l_base, const double* poses,
    const unsigned char* l_flag, const float* l_ele_l /*unused*/, int nl,
    int N,
    const float* r_des, const float* r_asa,
    const float* l_des, const float* l_asa,
    double* outS)
{
    // One static cache for the per-pose min buffers (sized to N×nr / N×nl).
    static int *dminR = 0; static float *dminL = 0;
    static int cap_nr = 0, cap_nl = 0, cap_N = 0;
    cudaError_t err;
#define CK(expr) do { err=(expr); if(err!=cudaSuccess) goto fail; } while(0)
    if (N > cap_N || nr > cap_nr || nl > cap_nl) {
        if (dminR) cudaFree(dminR);
        if (dminL) cudaFree(dminL);
        CK(cudaMalloc(&dminR,(size_t)N*nr*sizeof(int)));
        CK(cudaMalloc(&dminL,(size_t)N*nl*sizeof(float)));
        cap_nr=nr; cap_nl=nl; cap_N=N;
    }
    // Stage A
    CK(cudaMemset(dminR, 0x7F, (size_t)N*nr*sizeof(int)));   // +inf bits
    {
        int threads = 256;
        int bx = (nl + threads - 1) / threads;
        dim3 grid(bx, N);
        cpydock_min_kernel<<<grid, threads>>>(
            r_coords, r_flag, cell_start, cell_atoms,
            ncx, ncy, ncz, c_ox, c_oy, c_oz, c_sp,
            l_base, poses, l_flag, nr, nl, N, dminR, dminL);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    // Stage B (multi-block reduce into outS)
    CK(cudaMemset(outS, 0, (size_t)N*sizeof(double)));
    {
        int threads = 256;
        int bx = (nr + nl + threads - 1) / threads;
        if (bx < 8) bx = 8; if (bx > 64) bx = 64;
        dim3 grid(bx, N);
        cpydock_solv_kernel<<<grid, threads, threads*sizeof(float)>>>(
            dminR, dminL, r_asa, r_des, l_asa, l_des, nr, nl, N, outS);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    return 0;
fail:
    { const char* m=cudaGetErrorString(err); fprintf(stderr,"cuda_cpydock_solv error: %s\n",m); }
    return -1;
#undef CK
}
