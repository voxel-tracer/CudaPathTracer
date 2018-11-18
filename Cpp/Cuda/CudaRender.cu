#include "CudaRender.cuh"
#include "../Source/Config.h"
#include <stdlib.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__device__ float3 cRandomInUnitDisk(uint& state);

struct cRay
{
    __device__ cRay() {}
    __device__ cRay(const float3& orig_, const float3& dir_) : orig(orig_), dir(dir_), done(false) {}
    __device__ cRay(const cRay& r) : orig(r.orig), dir(r.dir), done(r.done) {}

    __device__ float3 pointAt(float t) const { return orig + dir * t; }

    float3 orig;
    float3 dir;
    bool done;
};

struct cSphere
{
    float3 center;
    float radius;

    __device__ float3 normalAt(const float3& pos) const { return (pos - center) / radius; }
};

struct cMaterial
{
    enum Type { Lambert, Metal, Dielectric };
    Type type;
    float3 albedo;
    float3 emissive;
    float roughness;
    float ri;
};

struct cCamera
{
    __device__ void GetRay(const float s, const float t, float3& ray_orig, float3& ray_dir, uint32_t& state) const
    {
        float3 rd = lensRadius * cRandomInUnitDisk(state);
        float3 offset = u * rd.x + v * rd.y;
        ray_orig = origin + offset;
        ray_dir = normalize(lowerLeftCorner + s * horizontal + t * vertical - origin - offset);
    }

    float3 origin;
    float3 lowerLeftCorner;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;
    float lensRadius;
};

struct DeviceData
{
    float *clr_x;
    float *clr_y;
    float *clr_z;
    float *atn_x;
    float *atn_y;
    float *atn_z;
    uint *ray_count;

    cCamera* camera;
    uint numRays;
    uint frame;
    uint width;
    uint height;
    uint samplesPerPixel;
    uint threadsPerBlock;
};

DeviceData deviceData;

const uint kSphereCount = 9;

__device__ __constant__ cSphere d_spheres[kSphereCount];
__device__ __constant__ cMaterial d_materials[kSphereCount];

__device__ float sqLength(const float3& v)
{
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

inline __device__ void AssertUnit(const float3& v)
{
    assert(fabsf(sqLength(v) - 1.0f) < 0.01f);
}

__device__ bool HitSphere(const cRay& r, const cSphere& s, float tMin, float tMax, float& outHitT)
{
    AssertUnit(r.dir);
    float3 oc = r.orig - s.center;
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - s.radius*s.radius;
    float discr = b * b - c;
    if (discr > 0)
    {
        float discrSq = sqrtf(discr);

        float t = (-b - discrSq);
        if (t < tMax && t > tMin)
        {
            outHitT = t;
            return true;
        }
        t = (-b + discrSq);
        if (t < tMax && t > tMin)
        {
            outHitT = t;
            return true;
        }
    }
    return false;
}

__device__ int hitWorld(const cRay& ray, float& closest, const float tMin, const float tMax)
{
    int hitId = -1;
    float hitT;

    closest = tMax;

    for (int i = 0; i < kSphereCount; ++i)
    {
        if (HitSphere(ray, d_spheres[i], tMin, closest, hitT))
        {
            closest = hitT;
            hitId = i;
        }
    }

    return hitId;
}

__device__ uint cXorShift32(uint& state)
{
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    state = x;
    return x;
}

__device__ float cRandomFloat01(uint& state)
{
    return (cXorShift32(state) & 0xFFFFFF) / 16777216.0f;
}

__device__ float3 cRandomInUnitDisk(uint& state)
{
    const float a = cRandomFloat01(state) * 2.0f * kPI;
    const float u = sqrtf(cRandomFloat01(state));
    return make_float3(u*cosf(a), u*sinf(a), 0);
}

float3 make_float3(const f3 f) {
    return make_float3(f.x, f.y, f.z);
}

__device__ float3 cRandomUnitVector(uint& state)
{
    float z = cRandomFloat01(state) * 2.0f - 1.0f;
    float a = cRandomFloat01(state) * 2.0f * kPI;
    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    return make_float3(x, y, z);
}

__device__ float3 cRandomInUnitSphere(uint& state)
{
    float z = cRandomFloat01(state) * 2.0f - 1.0f;
    float t = cRandomFloat01(state) * 2.0f * kPI;
    float r = sqrtf(fmaxf(0.0, 1.0f - z * z));
    float x = r * cosf(t);
    float y = r * sinf(t);
    float3 res = make_float3(x, y, z);
    res *= cbrtf(cRandomFloat01(state));
    return res;
}

/*
* based off http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
*/
__device__ uint cWang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ bool refract(const float3& v, const float3& n, float nint, float3& outRefracted)
{
    AssertUnit(v);
    float dt = dot(v, n);
    float discr = 1.0f - nint * nint*(1 - dt * dt);
    if (discr > 0)
    {
        outRefracted = nint * (v - n * dt) - n * sqrtf(discr);
        return true;
    }
    return false;
}

__device__ float cSchlick(float cosine, float ri)
{
    float r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    return r0 + (1 - r0)*powf(1 - cosine, 5);
}

__device__ bool ScatterNoLightSampling(const DeviceData& data, const cMaterial& mat, const cRay& r_in, const float hit_t, const int hit_id, float3& attenuation, cRay& scattered, uint& state)
{
    const float3 hitPos = r_in.pointAt(hit_t);
    const float3 hitNormal = d_spheres[hit_id].normalAt(hitPos);

    if (mat.type == cMaterial::Lambert)
    {
        // random point on unit sphere that is tangent to the hit point
        float3 target = hitPos + hitNormal + cRandomUnitVector(state);
        scattered = cRay(hitPos, normalize(target - hitPos));
        attenuation = mat.albedo;

        return true;
    }
    else if (mat.type == cMaterial::Metal)
    {
        AssertUnit(r_in.dir); AssertUnit(hitNormal);
        float3 refl = reflect(r_in.dir, hitNormal);
        // reflected ray, and random inside of sphere based on roughness
        float roughness = mat.roughness;
        scattered = cRay(hitPos, normalize(refl + roughness * cRandomInUnitSphere(state)));
        attenuation = mat.albedo;
        return dot(scattered.dir, hitNormal) > 0;
    }
    else if (mat.type == cMaterial::Dielectric)
    {
        AssertUnit(r_in.dir); AssertUnit(hitNormal);
        float3 outwardN;
        float3 rdir = r_in.dir;
        float3 refl = reflect(rdir, hitNormal);
        float nint;
        attenuation = make_float3(1, 1, 1);
        float3 refr;
        float reflProb;
        float cosine;
        if (dot(rdir, hitNormal) > 0)
        {
            outwardN = -1*hitNormal;
            nint = mat.ri;
            cosine = mat.ri * dot(rdir, hitNormal);
        }
        else
        {
            outwardN = hitNormal;
            nint = 1.0f / mat.ri;
            cosine = -dot(rdir, hitNormal);
        }
        if (refract(rdir, outwardN, nint, refr))
        {
            reflProb = cSchlick(cosine, mat.ri);
        }
        else
        {
            reflProb = 1;
        }
        if (cRandomFloat01(state) < reflProb)
            scattered = cRay(hitPos, normalize(refl));
        else
            scattered = cRay(hitPos, normalize(refr));
    }
    else
    {
        attenuation = make_float3(1, 0, 1);
        return false;
    }
    return true;
}

__device__ bool scatterHit(const cRay& ray, const int hit_id, const float hit_t, const uint depth, const DeviceData& data, uint& state, float3& color, float3& attenuation, cRay& scattered)
{
    const cMaterial mat = d_materials[hit_id];
    float3 local_attenuation;
    color += mat.emissive * attenuation;
    if (depth < kMaxDepth && ScatterNoLightSampling(data, mat, ray, hit_t, hit_id, local_attenuation, scattered, state))
    {
        attenuation *= local_attenuation;
        return true;
    }

    return false;
}

__device__ void scatterNoHit(const float3 ray_dir, float3& color, const float3& attenuation)
{
    // sky
    float t = 0.5f*(ray_dir.y + 1.0f);
    color += attenuation * ((1.0f - t)*make_float3(1) + t * make_float3(0.5f, 0.7f, 1.0f)) * 0.3f;
}

__device__ cRay generateRay(const uint x, const uint y, const DeviceData& data, uint& state)
{
    float u = float(x + cRandomFloat01(state)) / data.width;
    float v = float(y + cRandomFloat01(state)) / data.height;

    float3 ray_orig, ray_dir;
    data.camera->GetRay(u, v, ray_orig, ray_dir, state);

    return cRay(ray_orig, ray_dir);
}

__global__ void renderFrameKernel(const DeviceData data)
{
    const int rIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (rIdx >= data.numRays)
        return;

    const uint w = data.width*data.samplesPerPixel;
    const uint y = rIdx / w;
    const uint x = (rIdx % w) / data.samplesPerPixel;
    uint state = ((cWang_hash(rIdx) + (data.frame*kMaxDepth) * 101141101) * 336343633) | 1;

    cRay r = generateRay(x, y, data, state);

    float3 color = make_float3(
        data.clr_x[rIdx],
        data.clr_y[rIdx],
        data.clr_z[rIdx]
    );
    float3 attenuation = make_float3(1);
    bool ray_done = false;
    uint depth = 0;
    while (depth < kMaxDepth && !ray_done)
    {
        float hit_t;
        int hit_id = hitWorld(r, hit_t, kMinT, kMaxT);

        if (hit_id >= 0)
        {
            ray_done = !scatterHit(r, hit_id, hit_t, depth, data, state, color, attenuation, r);
        }
        else
        {
            // sky
            scatterNoHit(r.dir, color, attenuation);
            ray_done = true;
        }

        depth++;
    }

    data.clr_x[rIdx] = color.x;
    data.clr_y[rIdx] = color.y;
    data.clr_z[rIdx] = color.z;

    data.atn_x[rIdx] = attenuation.x;
    data.atn_y[rIdx] = attenuation.y;
    data.atn_z[rIdx] = attenuation.z;

    data.ray_count[rIdx] += depth;
}

void deviceInitData(const Camera* camera, const uint width, const uint height, const uint samplesPerPixel, const uint threadsPerBlock, const Sphere* spheres, const Material* materials, const int spheresCount, const int numRays)
{
    deviceData.numRays = numRays;
    deviceData.width = width;
    deviceData.height = height;
    deviceData.samplesPerPixel = samplesPerPixel;
    deviceData.threadsPerBlock = threadsPerBlock;

    gpuErrchk(cudaMalloc((void**)&deviceData.clr_x, numRays * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&deviceData.clr_y, numRays * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&deviceData.clr_z, numRays * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&deviceData.atn_x, numRays * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&deviceData.atn_y, numRays * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&deviceData.atn_z, numRays * sizeof(float)));

    gpuErrchk(cudaMalloc((void**)&deviceData.ray_count, numRays * sizeof(uint)));

    gpuErrchk(cudaMemsetAsync((void*)deviceData.clr_x, 0, numRays * sizeof(float)));
    gpuErrchk(cudaMemsetAsync((void*)deviceData.clr_y, 0, numRays * sizeof(float)));
    gpuErrchk(cudaMemsetAsync((void*)deviceData.clr_z, 0, numRays * sizeof(float)));
    gpuErrchk(cudaMemsetAsync((void*)deviceData.ray_count, 0, numRays * sizeof(uint)));

    gpuErrchk(cudaMalloc((void**)&deviceData.camera, sizeof(cCamera)));

    // copy spheres and materials to device
    gpuErrchk(cudaMemcpyToSymbol(d_spheres, spheres, kSphereCount * sizeof(cSphere)));
    gpuErrchk(cudaMemcpyToSymbol(d_materials, materials, kSphereCount * sizeof(cMaterial)));

    gpuErrchk(cudaMemcpy(deviceData.camera, camera, sizeof(cCamera), cudaMemcpyHostToDevice));
}

void deviceRenderFrame(const uint frame) {
    deviceData.frame = frame;
    // call kernel
    const uint blocksPerGrid = ceilf((float)deviceData.numRays / deviceData.threadsPerBlock);
    renderFrameKernel <<<blocksPerGrid, deviceData.threadsPerBlock >>> (deviceData);
}

void deviceEndRendering(f3* colors, unsigned long long& rayCount)
{
    const uint numRays = deviceData.numRays;

    float *f_tmp;
    gpuErrchk(cudaMallocHost((void**)&f_tmp, numRays * sizeof(float)));
    uint *i_tmp;
    gpuErrchk(cudaMallocHost((void**)&i_tmp, numRays * sizeof(uint)));

    // copy samples to host
    gpuErrchk(cudaMemcpy(f_tmp, deviceData.clr_x, numRays * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint i = 0; i < numRays; i++)
        colors[i].x = f_tmp[i];

    gpuErrchk(cudaMemcpy(f_tmp, deviceData.clr_y, numRays * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint i = 0; i < numRays; i++)
        colors[i].y = f_tmp[i];

    gpuErrchk(cudaMemcpy(f_tmp, deviceData.clr_z, numRays * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint i = 0; i < numRays; i++)
        colors[i].z = f_tmp[i];

    gpuErrchk(cudaMemcpy(i_tmp, deviceData.ray_count, numRays * sizeof(uint), cudaMemcpyDeviceToHost));
    for (uint i = 0; i < numRays; i++)
        rayCount += i_tmp[i];

    gpuErrchk(cudaFreeHost(f_tmp));
    gpuErrchk(cudaFreeHost(i_tmp));
}

void deviceFreeData()
{
    gpuErrchk(cudaFree(deviceData.clr_x));
    gpuErrchk(cudaFree(deviceData.clr_y));
    gpuErrchk(cudaFree(deviceData.clr_z));
    gpuErrchk(cudaFree(deviceData.atn_x));
    gpuErrchk(cudaFree(deviceData.atn_y));
    gpuErrchk(cudaFree(deviceData.atn_z));
    gpuErrchk(cudaFree(deviceData.ray_count));

    gpuErrchk(cudaFree(deviceData.camera));
}
