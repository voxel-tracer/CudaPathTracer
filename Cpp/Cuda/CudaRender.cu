#include "CudaRender.cuh"

inline float sqLength(const float3& v)
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

    const cRay& r = data.rays[rIdx];
    if (r.isDone())
        return;

    int hitId = -1;
    float closest = tMax, hitT;
    for (int i = 0; i < data.spheresCount; ++i)
    {
        if (HitSphere(r, data.spheres[i], tMin, closest, hitT))
        {
            closest = hitT;
            hitId = i;
        }
    }

    data.hits[rIdx] = cHit(closest, hitId);
}

void initDeviceData(const Sphere* spheres, const int spheresCount, const int numRays, DeviceData& data)
{
    data.numRays = numRays;
    data.spheresCount = spheresCount;

    // allocate device memory
    cudaMalloc((void**)&data.spheres, spheresCount * sizeof(cSphere));
    cudaMalloc((void**)&data.rays, numRays * sizeof(cRay));
    cudaMalloc((void**)&data.hits, numRays * sizeof(cHit));

    // copy spheres to device
    cudaMemcpy(data.spheres, spheres, spheresCount * sizeof(cSphere), cudaMemcpyHostToDevice);
}

void HitWorldDevice(const Ray* rays, float tMin, float tMax, Hit* hits, DeviceData data)
{
    // copy rays to device
    cudaMemcpy(data.rays, rays, data.numRays * sizeof(cRay), cudaMemcpyHostToDevice);

    // call kernel
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = ceilf((float)data.numRays / threadsPerBlock);

    HitWorldKernel <<<blocksPerGrid, threadsPerBlock >>> (data, tMin, tMax);

    // copy hits to host
    cudaMemcpy(hits, data.hits, data.numRays * sizeof(cHit), cudaMemcpyDeviceToHost);
}


void freeDeviceData(const DeviceData& data)
{
    cudaFree(data.spheres);
    cudaFree(data.rays);
    cudaFree(data.hits);
}
