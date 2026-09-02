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

#define NEAR2  100.0f        // 10 A squared (near cutoff)
#define LJ_CAP 1.0f
#define CP_W   6.0f
#define CP_F   0.75f
#define ES_CAP 0.01204819f   // 1.0*EPSILON/FACTOR with EPSILON=4, FACTOR=332

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
    double* __restrict__ out)                 // N * 2  (elec, vdw)
{
    int pose = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nl) return;
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
    if (qj != 0.f)
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
                        float ae = qj * r_ele[i] / d2;
                        if (ae > ES_CAP) ae = ES_CAP;
                        else if (ae < -ES_CAP) ae = -ES_CAP;
                        elec += ae;
                        float sv = svdwj * r_svdw[i];
                        float rr = vdwrj + r_vdwr[i];
                        float rr2 = rr * rr;
                        float p6 = rr2 * rr2 * rr2 / (d2 * d2 * d2);
                        float vp = sv * (p6*p6 - 2.0f*p6);
                        if (vp > LJ_CAP) vp = LJ_CAP;
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
    atomicAdd(&out[(size_t)pose * 2], (double)elec);
    atomicAdd(&out[(size_t)pose * 2 + 1], (double)vdw);
}

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
    double* out)
{
    cudaError_t err;
    float *d_phi=0,*d_rc=0,*d_re=0,*d_rsv=0,*d_rv=0; unsigned char *d_rh=0;
    int *d_cs=0,*d_ca=0;
    float *d_lb=0,*d_le=0,*d_lsv=0,*d_lv=0; unsigned char *d_lh=0;
    double *d_ps=0,*d_out=0;
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
    CK(cudaMalloc(&d_lb,(size_t)nl*3*sizeof(float)));
    CK(cudaMalloc(&d_ps,(size_t)N*7*sizeof(double)));
    CK(cudaMalloc(&d_le,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lsv,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lv,(size_t)nl*sizeof(float)));
    CK(cudaMalloc(&d_lh,(size_t)nl));
    CK(cudaMalloc(&d_out,(size_t)N*2*sizeof(double)));
    CK(cudaMemcpy(d_phi,phi,phi_b,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rc,r_coords,(size_t)nr*3*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_re,r_ele,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rsv,r_svdw,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rv,r_vdwr,(size_t)nr*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rh,r_heavy,(size_t)nr,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cs,cell_start,cb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ca,cell_atoms,(size_t)nr*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lb,l_base,(size_t)nl*3*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ps,poses,(size_t)N*7*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_le,l_ele,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lsv,l_svdw,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv,l_vdwr,(size_t)nl*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lh,l_heavy,(size_t)nl,cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out,0,(size_t)N*2*sizeof(double)));
    {
        int threads = 256;
        int bx = (nl + threads - 1) / threads;
        dim3 grid(bx, N);
        batch_full_score_kernel<<<grid, threads>>>(
            d_phi,nx,ny,nz,ox,oy,oz,sp,
            d_rc,d_re,d_rsv,d_rv,d_rh,d_cs,d_ca,ncx,ncy,ncz,c_ox,c_oy,c_oz,c_sp,
            d_lb,d_ps,d_le,d_lsv,d_lv,d_lh,nl,N,d_out);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpy(out,d_out,(size_t)N*2*sizeof(double),cudaMemcpyDeviceToHost));
    cudaFree(d_phi);cudaFree(d_rc);cudaFree(d_re);cudaFree(d_rsv);cudaFree(d_rv);cudaFree(d_rh);
    cudaFree(d_cs);cudaFree(d_ca);cudaFree(d_lb);cudaFree(d_ps);cudaFree(d_le);cudaFree(d_lsv);cudaFree(d_lv);cudaFree(d_lh);
    cudaFree(d_out);
    return 0;
fail:
    { const char* m=cudaGetErrorString(err); fprintf(stderr,"cuda_batch_score error: %s\n",m); }
    if(d_phi)cudaFree(d_phi);if(d_rc)cudaFree(d_rc);if(d_re)cudaFree(d_re);if(d_rsv)cudaFree(d_rsv);
    if(d_rv)cudaFree(d_rv);if(d_rh)cudaFree(d_rh);if(d_cs)cudaFree(d_cs);if(d_ca)cudaFree(d_ca);
    if(d_lb)cudaFree(d_lb);if(d_ps)cudaFree(d_ps);if(d_le)cudaFree(d_le);if(d_lsv)cudaFree(d_lsv);if(d_lv)cudaFree(d_lv);
    if(d_lh)cudaFree(d_lh);if(d_out)cudaFree(d_out);
    return -1;
#undef CK
}
