#pragma once

#include <helper_math.h>
#include <assert.h>

#include "../Source/Maths.h"
#include "device_launch_parameters.h"

struct cHit
{
    __device__ cHit() {}
    __device__ cHit(float _t, float _id) :t(_t), id(_id) {}

    float t;
    int id;
};

struct cRay
{
    __device__ cRay() {}
    __device__ cRay(const float3& orig_, const float3& dir_) : orig(orig_), dir(dir_) {}

    __device__ float3 pointAt(float t) const { return orig + dir * t; }
    __device__ bool isDone() const { return dir.x == 0 && dir.y == 0 && dir.z == 0; }

    float3 orig;
    float3 dir;
};

struct cSphere
{
    float3 center;
    float radius;
    float _not_used;

    __device__ float3 normalAt(const float3& pos) const { return (pos - center) / radius; }
};

struct DeviceData
{
    cRay* rays;
    cHit* hits;
    cSphere* spheres;
    int numRays;
    int spheresCount;
};

void initDeviceData(const Sphere* spheres, const int spheresCount, const int numRays, DeviceData& data);

void HitWorldDevice(const Ray* rays, float tMin, float tMax, Hit* hits, DeviceData data);

void freeDeviceData(const DeviceData& data);
