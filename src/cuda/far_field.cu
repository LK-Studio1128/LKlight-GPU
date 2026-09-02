// far_field CUDA kernel + host wrapper for LKlight-CUDA.
// Computes  E_far = Σ_j q_j · φ(x_j)  where φ is the receptor far-field grid
// (10-30 Å electrostatics, 1 Å spacing) built on the host (grid_dna.rs).
// One thread per ligand atom → trilinear gather → atomic reduction.
//
// Compile (from build.rs when --features cuda is set):
//   nvcc -O3 -arch=native -c src/cuda/far_field.cu -o target/cuda/far_field.o
// then archive into a static lib linked by cargo.

#include <cuda_runtime.h>
#include <stdio.h>

__device__ __forceinline__ float sample_phi(
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
    float c000 = phi[base];
    float c100 = phi[base + 1];
    float c010 = phi[base + nx];
    float c110 = phi[base + nx + 1];
    float c001 = phi[base + (size_t)ny * nx];
    float c101 = phi[base + (size_t)ny * nx + 1];
    float c011 = phi[base + (size_t)ny * nx + nx];
    float c111 = phi[base + (size_t)ny * nx + nx + 1];
    float c00 = c000 * (1.f - tx) + c100 * tx;
    float c10 = c010 * (1.f - tx) + c110 * tx;
    float c01 = c001 * (1.f - tx) + c101 * tx;
    float c11 = c011 * (1.f - tx) + c111 * tx;
    float c0 = c00 * (1.f - ty) + c10 * ty;
    float c1 = c01 * (1.f - ty) + c11 * ty;
    return c0 * (1.f - tz) + c1 * tz;
}

__global__ void far_field_kernel(
    const float* __restrict__ phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* __restrict__ coords,   // n × 3 (x,y,z)
    const float* __restrict__ charges,  // n
    int n, double* __restrict__ out)    // out[0] atomic-accumulated
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float q = charges[j];
    if (q == 0.f) return;
    float v = sample_phi(phi, nx, ny, nz, ox, oy, oz, sp,
                         coords[j*3], coords[j*3+1], coords[j*3+2]);
    if (v != 0.f) atomicAdd(out, (double)(q * v));
}

extern "C" int cuda_far_field(
    const float* phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float* coords, const float* charges, int n,
    double* result)
{
    float *d_phi = nullptr, *d_coords = nullptr, *d_charges = nullptr;
    double *d_out = nullptr;
    cudaError_t err;
    size_t phi_bytes = (size_t)nx * ny * nz * sizeof(float);
    size_t c_bytes   = (size_t)n * sizeof(float);

    err = cudaMalloc((void**)&d_phi, phi_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&d_coords, (size_t)n * 3 * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&d_charges, c_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&d_out, sizeof(double));
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(d_phi, phi, phi_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(d_coords, coords, (size_t)n * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(d_charges, charges, c_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemset(d_out, 0, sizeof(double));
    if (err != cudaSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        far_field_kernel<<<blocks, threads>>>(d_phi, nx, ny, nz, ox, oy, oz, sp,
                                              d_coords, d_charges, n, d_out);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMemcpy(result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(d_phi); cudaFree(d_coords); cudaFree(d_charges); cudaFree(d_out);
    return 0;

fail:
    {
        const char* msg = cudaGetErrorString(err);
        fprintf(stderr, "cuda_far_field error: %s\n", msg);
    }
    if (d_phi) cudaFree(d_phi);
    if (d_coords) cudaFree(d_coords);
    if (d_charges) cudaFree(d_charges);
    if (d_out) cudaFree(d_out);
    return -1;
}
