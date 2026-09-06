// lk_metal.m — LKlight-GPU Apple GPU (Metal) batch scorer, C shim.
//
// Compiled into a static library (src/metal/lk_metal.m -> liblkmetal.a) by
// build.rs when the `metal` cargo feature is enabled on macOS. The MSL kernel
// below is a verbatim port of the validated POC kernel
// (functest/ht1000_mac/metal_poc/run_metal_v4_tg128_final.m):
//   * one threadgroup per pose, `tg` threads striding over ligand atoms;
//   * far:  trilinear gather of the 0.5 A receptor field phi, x q_j;
//   * near: 27-cell (10 A) receptor cell list — clamped electrostatics
//           (|q_i q_j|/d^2 capped at 4/332), capped LJ (<= 1.0) and the
//           heavy-atom clash penalty (6.0 . (0.75 . sum r_vdw - d));
//   * poses arrive CPU f64 pre-transformed (host side, Rust) as pose-major
//     float3 and are memcpy'd into a shared MTLBuffer (zero-copy to the GPU).
//
// The C entry points mirror the CUDA shim in src/cuda/*.cu so the Rust host
// (src/metal_score.rs) looks exactly like src/gpu_score.rs.
//
// All static receptor/ligand/field data is COPIED into Metal buffers at
// context creation; the context persists across GSO steps (created once per
// DNA instance) and only the per-step transformed coordinates change.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static const char *LK_MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant float NEAR2 = 100.0f;\n"
"constant float LJ_CAP = 1.0f;\n"
"constant float CP_W = 6.0f;\n"
"constant float CP_F = 0.75f;\n"
"constant float ES_CAP = 0.01204819f;\n"
"// Family flags (buffer 11 sizes[3]): LJ always on. FAR=1 ELEC=2 CLASH=4.\n"
"inline float sample_phi(device const float* phi, int nx, int ny, int nz,\n"
"                        float ox, float oy, float oz, float sp,\n"
"                        float x, float y, float z) {\n"
"    float fx=(x-ox)/sp, fy=(y-oy)/sp, fz=(z-oz)/sp;\n"
"    if (fx<0.f||fy<0.f||fz<0.f) return 0.f;\n"
"    int ix=(int)fx, iy=(int)fy, iz=(int)fz;\n"
"    if (ix+1>=nx||iy+1>=ny||iz+1>=nz) return 0.f;\n"
"    float tx=fx-ix, ty=fy-iy, tz=fz-iz;\n"
"    uint base=(uint)iz*(uint)ny*(uint)nx+(uint)iy*(uint)nx+(uint)ix;\n"
"    float c000=phi[base], c100=phi[base+1], c010=phi[base+nx], c110=phi[base+nx+1];\n"
"    float c001=phi[base+(uint)ny*nx], c101=phi[base+(uint)ny*nx+1];\n"
"    float c011=phi[base+(uint)ny*nx+nx], c111=phi[base+(uint)ny*nx+nx+1];\n"
"    float c00=c000*(1.f-tx)+c100*tx, c10=c010*(1.f-tx)+c110*tx;\n"
"    float c01=c001*(1.f-tx)+c101*tx, c11=c011*(1.f-tx)+c111*tx;\n"
"    float c0=c00*(1.f-ty)+c10*ty, c1=c01*(1.f-ty)+c11*ty;\n"
"    return c0*(1.f-tz)+c1*tz;\n"
"}\n"
"// Kahan-compensated add: acc=(sum,comp); returns updated pair.\n"
"inline float2 kadd(float2 a, float x) {\n"
"    float s=a.x+x, bb=s-a.x, e=(a.x-(s-bb))+(x-bb);\n"
"    return float2(s, a.y+e);\n"
"}\n"
"kernel void batch_score(\n"
"    device const float* phi [[buffer(0)]],\n"
"    device const int* fdims [[buffer(1)]],\n"
"    device const float* fgeo [[buffer(2)]],\n"
"    device const float4* rpk [[buffer(3)]],\n"
"    device const float4* rpp [[buffer(4)]],\n"
"    device const int* cell_start [[buffer(5)]],\n"
"    device const int* cell_atoms [[buffer(6)]],\n"
"    device const int* cgeo_i [[buffer(7)]],\n"
"    device const float* cgeo_f [[buffer(8)]],\n"
"    device const float* lc [[buffer(9)]],\n"
"    device const float4* lpp [[buffer(10)]],\n"
"    device const int* sizes [[buffer(11)]],\n"
"    device float* outE [[buffer(12)]],\n"
"    device float* outV [[buffer(13)]],\n"
"    uint pose [[threadgroup_position_in_grid]],\n"
"    uint lid [[thread_position_in_threadgroup]],\n"
"    uint tg_sz [[threads_per_threadgroup]])\n"
"{\n"
"    const int nl = sizes[1];\n"
"    const uint flags = (uint)sizes[3];\n"
"    float2 elec=float2(0.f,0.f), vdw=float2(0.f,0.f);\n"
"    for (int j=(int)lid; j<nl; j+=(int)tg_sz) {\n"
"        const int nx=fdims[0], ny=fdims[1], nz=fdims[2];\n"
"        const float ox=fgeo[0], oy=fgeo[1], oz=fgeo[2], sp=fgeo[3];\n"
"        const int nr=sizes[0];\n"
"        const int ncx=cgeo_i[0], ncy=cgeo_i[1], ncz=cgeo_i[2];\n"
"        const float c_ox=cgeo_f[0], c_oy=cgeo_f[1], c_oz=cgeo_f[2], c_sp=cgeo_f[3];\n"
"        const uint p3 = pose*(uint)nl*3u + (uint)j*3u;\n"
"        float x=lc[p3], y=lc[p3+1], z=lc[p3+2];\n"
"        float4 lq=lpp[j];\n"
"        float qj=lq.x, svdwj=lq.y, vdwrj=lq.z; bool hj = lq.w>0.5f;\n"
"        if ((flags & 1u) && qj!=0.f) {\n"
"            float2 t=kadd(elec, qj*sample_phi(phi,nx,ny,nz,ox,oy,oz,sp,x,y,z)); elec=t;\n"
"        }\n"
"        int cxi=(int)floor((x-c_ox)/c_sp), cyi=(int)floor((y-c_oy)/c_sp), czi=(int)floor((z-c_oz)/c_sp);\n"
"        for (int dz=-1; dz<=1; dz++) {\n"
"            int czz=czi+dz; if (czz<0||czz>=ncz) continue;\n"
"            for (int dy=-1; dy<=1; dy++) {\n"
"                int cyy=cyi+dy; if (cyy<0||cyy>=ncy) continue;\n"
"                for (int dx=-1; dx<=1; dx++) {\n"
"                    int cxx=cxi+dx; if (cxx<0||cxx>=ncx) continue;\n"
"                    int cell=(czz*ncy+cyy)*ncx+cxx;\n"
"                    int b0=cell_start[cell], e=cell_start[cell+1];\n"
"                    for (int p=b0; p<e; p++) {\n"
"                        int i=cell_atoms[p];\n"
"                        float4 rA=rpk[i];\n"
"                        float dxf=x-rA.x, dyf=y-rA.y, dzf=z-rA.z;\n"
"                        float d2=dxf*dxf+dyf*dyf+dzf*dzf;\n"
"                        if (d2<=NEAR2) {\n"
"                            float4 rB=rpp[i];\n"
"                            if (flags & 2u) {\n"
"                                float ae=qj*rA.w/d2;\n"
"                                if (ae>ES_CAP) ae=ES_CAP;\n"
"                                if (ae<-ES_CAP) ae=-ES_CAP;\n"
"                                float2 t=kadd(elec,ae); elec=t;\n"
"                            }\n"
"                            float sv=svdwj*rB.x;\n"
"                            float rr=vdwrj+rB.y;\n"
"                            float p6=(rr*rr)*(rr*rr)*(rr*rr)/(d2*d2*d2);\n"
"                            float vp=sv*(p6*p6-2.f*p6);\n"
"                            if (vp>LJ_CAP) vp=LJ_CAP;\n"
"                            if ((flags & 4u) && hj && rB.z>0.5f) {\n"
"                                float dmin=CP_F*rr;\n"
"                                if (d2<dmin*dmin) vp+=CP_W*(dmin-sqrt(d2));\n"
"                            }\n"
"                            float2 u=kadd(vdw,vp); vdw=u;\n"
"                        }\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup float2 se[512], sv[512];\n"
"    se[lid]=elec; sv[lid]=vdw;\n"
"    for (uint st=tg_sz/2; st>0; st>>=1) {\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (lid<st) {\n"
"            float2 a=se[lid], b=se[lid+st];\n"
"            float s=a.x+b.x, bb=s-a.x, err=(a.x-(s-bb))+(b.x-bb);\n"
"            se[lid]=float2(s,a.y+b.y+err);\n"
"            float2 c=sv[lid], d=sv[lid+st];\n"
"            float s2=c.x+d.x, bb2=s2-c.x, err2=(c.x-(s2-bb2))+(d.x-bb2);\n"
"            sv[lid]=float2(s2,c.y+d.y+err2);\n"
"        }\n"
"    }\n"
"    if (lid==0) { outE[pose]=se[0].x+se[0].y; outV[pose]=sv[0].x+sv[0].y; }\n"
"}\n";

// ---------------------------------------------------------------------------
// C API (declared in metal_score.rs)
// ---------------------------------------------------------------------------
// int  lk_metal_available(void);
// void *lk_metal_ctx_create(phi, nx,ny,nz, ox,oy,oz,sp,
//                           r_coords, r_ele, r_svdw, r_vdwr, r_heavy, nr,
//                           cell_start, cell_atoms, ncx,ncy,ncz,
//                           c_ox,c_oy,c_oz,c_sp,
//                           l_ele, l_svdw, l_vdwr, l_heavy, nl, tg);
// int  lk_metal_score(void *ctx, const float *lc, int n_pose,
//                     float *outE, float *outV);
// void lk_metal_ctx_destroy(void *ctx);

typedef struct LkMetalCtx {
    id<MTLDevice> __strong device;
    id<MTLLibrary> __strong lib;
    id<MTLComputePipelineState> __strong pipe;
    id<MTLCommandQueue> __strong queue;
    id<MTLBuffer> __strong bphi, bfd, bfg, brk, brp, bcs, bca, bci, bcf, blc, blp, bsz, bE, bV;
    int tg;
    int nl, nr;
    int cap;      // current lc/out capacity in poses
} LkMetalCtx;

int lk_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d != nil;
    }
}

void *lk_metal_ctx_create(
    const float *phi, int nx, int ny, int nz,
    float ox, float oy, float oz, float sp,
    const float *r_coords, const float *r_ele, const float *r_svdw,
    const float *r_vdwr, const unsigned char *r_heavy, int nr,
    const int *cell_start, const int *cell_atoms,
    int ncx, int ncy, int ncz,
    float c_ox, float c_oy, float c_oz, float c_sp,
    const float *l_ele, const float *l_svdw, const float *l_vdwr,
    const unsigned char *l_heavy, int nl,
    int tg, int mode) {
    @autoreleasepool {
        NSError *err = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;
        id<MTLLibrary> lib = [device newLibraryWithSource:
                              [NSString stringWithUTF8String:LK_MSL]
                              options:nil error:&err];
        if (!lib) { fprintf(stderr, "[metal] MSL compile failed: %s\n",
                            err.localizedDescription.UTF8String); return NULL; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"batch_score"];
        if (!fn) { fprintf(stderr, "[metal] batch_score function missing\n"); return NULL; }
        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "[metal] pipeline: %s\n",
                             err.localizedDescription.UTF8String); return NULL; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (tg < 2 || (tg & (tg - 1)) != 0) tg = 128;   // 2^N only (tree reduce)
        if (tg > 512) tg = 512;

        int phiN = nx * ny * nz;
        int csN = (ncx + 1) * (ncy + 1) * (ncz + 1) + 1;   // cell_start incl. sentinel
        if (nr <= 0 || nl <= 0 || csN <= 0) return NULL;
        // VDW family has no far field (flags clear FAR): hand a 1-element
        // placeholder grid — the kernel never samples phi without the FAR flag.
        float zphi = 0.f;
        if (phiN <= 0) { nx = ny = nz = 1; phiN = 1; phi = &zphi; }

        LkMetalCtx *c = calloc(1, sizeof(LkMetalCtx));
        if (!c) return NULL;
        c->device = device; c->lib = lib; c->pipe = pipe; c->queue = queue;
        c->tg = tg; c->nl = nl; c->nr = nr; c->cap = 0;

        // Pack receptor attributes into AoS float4 arrays:
        //   rpk[i] = (x, y, z, q);  rpp[i] = (svdw, vdwr, heavy, 0)
        NSMutableData *rpkD = [NSMutableData dataWithLength:(NSUInteger)nr * 16];
        NSMutableData *rppD = [NSMutableData dataWithLength:(NSUInteger)nr * 16];
        {
            float *ok = rpkD.mutableBytes, *op = rppD.mutableBytes;
            for (int i = 0; i < nr; i++) {
                ok[i*4]   = r_coords[i*3];
                ok[i*4+1] = r_coords[i*3+1];
                ok[i*4+2] = r_coords[i*3+2];
                ok[i*4+3] = r_ele[i];
                op[i*4]   = r_svdw[i];
                op[i*4+1] = r_vdwr[i];
                op[i*4+2] = (float)r_heavy[i];
                op[i*4+3] = 0.f;
            }
        }
        // lpp[j] = (q, svdw, vdwr, heavy)
        NSMutableData *lppD = [NSMutableData dataWithLength:(NSUInteger)nl * 16];
        {
            float *ol = lppD.mutableBytes;
            for (int j = 0; j < nl; j++) {
                ol[j*4]   = l_ele[j];
                ol[j*4+1] = l_svdw[j];
                ol[j*4+2] = l_vdwr[j];
                ol[j*4+3] = (float)l_heavy[j];
            }
        }

        int32_t fdims[3] = {nx, ny, nz};
        float fgeo[4] = {ox, oy, oz, sp};
        int32_t cgi[3] = {ncx, ncy, ncz};
        float cgf[4] = {c_ox, c_oy, c_oz, c_sp};
        int32_t sizes[4] = {nr, nl, 0, mode};   // [3] = per-family flags

        c->bphi = [device newBufferWithBytes:phi length:(NSUInteger)phiN * 4
                                     options:MTLResourceStorageModeShared];
        c->brk = [device newBufferWithBytes:rpkD.bytes length:rpkD.length
                                    options:MTLResourceStorageModeShared];
        c->brp = [device newBufferWithBytes:rppD.bytes length:rppD.length
                                    options:MTLResourceStorageModeShared];
        c->bcs = [device newBufferWithBytes:cell_start length:(NSUInteger)csN * 4
                                    options:MTLResourceStorageModeShared];
        c->bca = [device newBufferWithBytes:cell_atoms length:(NSUInteger)nr * 4
                                    options:MTLResourceStorageModeShared];
        c->blp = [device newBufferWithBytes:lppD.bytes length:lppD.length
                                    options:MTLResourceStorageModeShared];
        c->bfd = [device newBufferWithBytes:fdims length:12 options:MTLResourceStorageModeShared];
        c->bfg = [device newBufferWithBytes:fgeo length:16 options:MTLResourceStorageModeShared];
        c->bci = [device newBufferWithBytes:cgi length:12 options:MTLResourceStorageModeShared];
        c->bcf = [device newBufferWithBytes:cgf length:16 options:MTLResourceStorageModeShared];
        c->bsz = [device newBufferWithBytes:sizes length:16 options:MTLResourceStorageModeShared];
        return c;
    }
}

static int ensure_buffers(LkMetalCtx *c, int n_pose) {
    if (n_pose <= c->cap && c->blc && c->bE && c->bV) return 0;
    int need = n_pose > 32 ? n_pose : 32;   // small headroom; engine batches ~1000
    c->blc = [c->device newBufferWithLength:(NSUInteger)need * (NSUInteger)c->nl * 12
                                     options:MTLResourceStorageModeShared];
    c->bE = [c->device newBufferWithLength:(NSUInteger)need * 4
                                   options:MTLResourceStorageModeShared];
    c->bV = [c->device newBufferWithLength:(NSUInteger)need * 4
                                   options:MTLResourceStorageModeShared];
    if (!c->blc || !c->bE || !c->bV) return -1;
    c->cap = need;
    return 0;
}

int lk_metal_score(void *vctx, const float *lc, int n_pose,
                   float *outE, float *outV) {
    LkMetalCtx *c = (LkMetalCtx *)vctx;
    if (!c || n_pose <= 0) return -1;
    @autoreleasepool {
        if (ensure_buffers(c, n_pose) != 0) return -1;
        int32_t *szm = (int32_t *)c->bsz.contents;
        szm[2] = n_pose;
        // sizes[3] keeps the family flags set at ctx_create
        memcpy(c->blc.contents, lc, (size_t)n_pose * (size_t)c->nl * 12);
        MTLSize groups = MTLSizeMake((NSUInteger)n_pose, 1, 1);
        MTLSize threads = MTLSizeMake((NSUInteger)c->tg, 1, 1);
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:c->pipe];
        id<MTLBuffer> bufs[14] = {c->bphi, c->bfd, c->bfg, c->brk, c->brp,
                                  c->bcs, c->bca, c->bci, c->bcf, c->blc,
                                  c->blp, c->bsz, c->bE, c->bV};
        for (int i = 0; i < 14; i++) [e setBuffer:bufs[i] offset:0 atIndex:i];
        [e dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        float *eP = c->bE.contents, *vP = c->bV.contents;
        if (outE) memcpy(outE, eP, (size_t)n_pose * 4);
        if (outV) memcpy(outV, vP, (size_t)n_pose * 4);
        return 0;
    }
}

void lk_metal_ctx_destroy(void *vctx) {
    if (!vctx) return;
    LkMetalCtx *c = (LkMetalCtx *)vctx;
    @autoreleasepool {
        // All ObjC ivars are strong references; releasing the pool of the
        // owning autorelease scope is not enough for ivars created outside it,
        // so nil them out (ARC releases) before free.
        c->device = nil; c->lib = nil; c->pipe = nil; c->queue = nil;
        c->bphi = nil; c->bfd = nil; c->bfg = nil; c->brk = nil; c->brp = nil;
        c->bcs = nil; c->bca = nil; c->bci = nil; c->bcf = nil; c->blc = nil;
        c->blp = nil; c->bsz = nil; c->bE = nil; c->bV = nil;
    }
    free(c);
}
