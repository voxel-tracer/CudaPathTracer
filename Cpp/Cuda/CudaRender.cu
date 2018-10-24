#include "CudaRender.cuh"
#include "../Source/Config.h"

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
    float *rays_orig_x;
    float *rays_orig_y;
    float *rays_orig_z;
    float *rays_dir_x;
    float *rays_dir_y;
    float *rays_dir_z;
    bool *rays_done;

    float *hits_t;
    int *hits_id;

    float *clr_x;
    float *clr_y;
    float *clr_z;
    float *atn_x;
    float *atn_y;
    float *atn_z;

    float *h_tmp;

    cCamera* camera;
    uint numRays;
    uint frame;
    uint width;
    uint height;
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

__global__ void HitWorldKernel(const DeviceData data, float tMin, float tMax)
{
    const int rIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (rIdx >= data.numRays)
        return;

    if (data.rays_done[rIdx])
        return;

    const float3 ray_orig = make_float3(
        data.rays_orig_x[rIdx],
        data.rays_orig_y[rIdx],
        data.rays_orig_z[rIdx]);
    const float3 ray_dir = make_float3(
        data.rays_dir_x[rIdx],
        data.rays_dir_y[rIdx],
        data.rays_dir_z[rIdx]);

    const cRay r(ray_orig, ray_dir);

    int hitId = -1;
    float closest = tMax, hitT;
    for (int i = 0; i < kSphereCount; ++i)
    {
        if (HitSphere(r, d_spheres[i], tMin, closest, hitT))
        {
            closest = hitT;
            hitId = i;
        }
    }

    data.hits_t[rIdx] = closest;
    data.hits_id[rIdx] = hitId;
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
    float3 p;
    do
    {
        p = make_float3(2 * cRandomFloat01(state) - 1, 2 * cRandomFloat01(state) - 1, 0);
    } while (dot(p, p) >= 1.0);
    return p;
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
    float3 p;
    do {
        p = make_float3(2*cRandomFloat01(state) - 1, 2*cRandomFloat01(state) - 1, 2*cRandomFloat01(state) - 1);
    } while (sqLength(p) >= 1.0);
    return p;
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

__global__ void ScatterKernel(const DeviceData data, const uint depth)
{
    const int rIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (rIdx >= data.numRays)
        return;

    if (data.rays_done[rIdx])
        return;

    const float3 ray_orig = make_float3(
        data.rays_orig_x[rIdx],
        data.rays_orig_y[rIdx],
        data.rays_orig_z[rIdx]);
    const float3 ray_dir = make_float3(
        data.rays_dir_x[rIdx],
        data.rays_dir_y[rIdx],
        data.rays_dir_z[rIdx]);

    const cRay r(ray_orig, ray_dir);

    uint state = (cWang_hash(rIdx) + (data.frame*kMaxDepth + depth) * 101141101) * 336343633;

    float3 color = make_float3(
        data.clr_x[rIdx],
        data.clr_y[rIdx],
        data.clr_z[rIdx]
    );
    float3 attenuation;
    if (depth == 0) {
        attenuation = make_float3(1);
    }
    else {
        attenuation = make_float3(
            data.atn_x[rIdx],
            data.atn_y[rIdx],
            data.atn_z[rIdx]
        );
    }

    const int hit_id = data.hits_id[rIdx];
    if (hit_id >= 0)
    {
        const float hit_t = data.hits_t[rIdx];
        cRay scattered;
        const cMaterial mat = d_materials[hit_id];
        float3 local_attenuation;
        color += mat.emissive * attenuation;
        if (depth < kMaxDepth && ScatterNoLightSampling(data, mat, r, hit_t, hit_id, local_attenuation, scattered, state))
        {
            attenuation *= local_attenuation;
            data.rays_orig_x[rIdx] = scattered.orig.x;
            data.rays_orig_y[rIdx] = scattered.orig.y;
            data.rays_orig_z[rIdx] = scattered.orig.z;
            data.rays_dir_x[rIdx] = scattered.dir.x;
            data.rays_dir_y[rIdx] = scattered.dir.y;
            data.rays_dir_z[rIdx] = scattered.dir.z;
        }
        else
        {
            data.rays_done[rIdx] = true;
        }
    }
    else
    {
        // sky
        float3 unitDir = r.dir;
        float t = 0.5f*(unitDir.y + 1.0f);
        color += attenuation * ((1.0f - t)*make_float3(1) + t * make_float3(0.5f, 0.7f, 1.0f)) * 0.3f;
        data.rays_done[rIdx] = true;
    }

    data.clr_x[rIdx] = color.x;
    data.clr_y[rIdx] = color.y;
    data.clr_z[rIdx] = color.z;

    //TODO no need to write this in the last depth iteration
    data.atn_x[rIdx] = attenuation.x;
    data.atn_y[rIdx] = attenuation.y;
    data.atn_z[rIdx] = attenuation.z;
}

__global__ void generateRays(const DeviceData data)
{
    const int rIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (rIdx >= data.numRays)
        return;

    const uint w = data.width*DO_SAMPLES_PER_PIXEL;
    const uint y = rIdx / w;
    const uint x = (rIdx % w) / DO_SAMPLES_PER_PIXEL;
    uint state = ((cWang_hash(rIdx) + (data.frame*kMaxDepth) * 101141101) * 336343633) | 1;

    float u = float(x + cRandomFloat01(state)) / data.width;
    float v = float(y + cRandomFloat01(state)) / data.height;

    float3 ray_orig, ray_dir;
    data.camera->GetRay(u, v, ray_orig, ray_dir, state);

    data.rays_orig_x[rIdx] = ray_orig.x;
    data.rays_orig_y[rIdx] = ray_orig.y;
    data.rays_orig_z[rIdx] = ray_orig.z;
    data.rays_dir_x[rIdx] = ray_dir.x;
    data.rays_dir_y[rIdx] = ray_dir.y;
    data.rays_dir_z[rIdx] = ray_dir.z;
    data.rays_done[rIdx] = false; //TODO just use memset
}

void deviceInitData(const Camera* camera, const uint width, const uint height, const Sphere* spheres, const Material* materials, const int spheresCount, const int numRays)
{
    deviceData.numRays = numRays;
    deviceData.width = width;
    deviceData.height = height;

    cudaMallocHost((void**)&deviceData.h_tmp, numRays * sizeof(float));

    // allocate device memory
    cudaMalloc((void**)&deviceData.rays_orig_x, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_orig_y, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_orig_z, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_dir_x, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_dir_y, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_dir_z, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.rays_done, numRays * sizeof(bool));

    cudaMalloc((void**)&deviceData.hits_t, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.hits_id, numRays * sizeof(int));

    cudaMalloc((void**)&deviceData.clr_x, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.clr_y, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.clr_z, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.atn_x, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.atn_y, numRays * sizeof(float));
    cudaMalloc((void**)&deviceData.atn_z, numRays * sizeof(float));

    cudaMemsetAsync((void*)deviceData.clr_x, 0, numRays * sizeof(float));
    cudaMemsetAsync((void*)deviceData.clr_y, 0, numRays * sizeof(float));
    cudaMemsetAsync((void*)deviceData.clr_z, 0, numRays * sizeof(float));

    cudaMalloc((void**)&deviceData.camera, sizeof(cCamera));

    // copy spheres and materials to device
    cudaMemcpyToSymbol(d_spheres, spheres, kSphereCount * sizeof(cSphere));
    cudaMemcpyToSymbol(d_materials, materials, kSphereCount * sizeof(cMaterial));

    cudaMemcpy(deviceData.camera, camera, sizeof(cCamera), cudaMemcpyHostToDevice);
}

void deviceStartFrame(const uint frame) {
    deviceData.frame = frame;

    // call kernel
    const int blocksPerGrid = ceilf((float)deviceData.numRays / kThreadsPerBlock);
    generateRays <<<blocksPerGrid, kThreadsPerBlock >>> (deviceData);
}

void deviceRenderFrame(const float tMin, const float tMax, const uint depth)
{
    // call kernel
    const int blocksPerGrid = ceilf((float)deviceData.numRays / kThreadsPerBlock);

    HitWorldKernel <<<blocksPerGrid, kThreadsPerBlock >> > (deviceData, tMin, tMax);
    ScatterKernel <<<blocksPerGrid, kThreadsPerBlock >> > (deviceData, depth);
}

void deviceEndRendering(f3* colors)
{
    const uint numRays = deviceData.numRays;

    // copy samples to host
    cudaMemcpy(deviceData.h_tmp, deviceData.clr_x, numRays * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < numRays; i++)
        colors[i].x = deviceData.h_tmp[i];

    cudaMemcpy(deviceData.h_tmp, deviceData.clr_y, numRays * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < numRays; i++)
        colors[i].y = deviceData.h_tmp[i];

    cudaMemcpy(deviceData.h_tmp, deviceData.clr_z, numRays * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < numRays; i++)
        colors[i].z = deviceData.h_tmp[i];
}

void deviceFreeData()
{
    cudaFree(deviceData.rays_orig_x);
    cudaFree(deviceData.rays_orig_y);
    cudaFree(deviceData.rays_orig_z);
    cudaFree(deviceData.rays_dir_x);
    cudaFree(deviceData.rays_dir_y);
    cudaFree(deviceData.rays_dir_z);
    cudaFree(deviceData.rays_done);

    cudaFree(deviceData.hits_t);
    cudaFree(deviceData.hits_id);

    cudaFree(deviceData.clr_x);
    cudaFree(deviceData.clr_y);
    cudaFree(deviceData.clr_z);
    cudaFree(deviceData.atn_x);
    cudaFree(deviceData.atn_y);
    cudaFree(deviceData.atn_z);
    cudaFree(deviceData.h_tmp);

    cudaFree(deviceData.camera);
}
