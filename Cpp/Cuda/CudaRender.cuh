#pragma once

#include <helper_math.h>
#include <assert.h>

#include "../Source/Maths.h"
#include "device_launch_parameters.h"

struct cHit
{
    float3 pos;
    float3 normal;
    float t;
    int id = -1;
};

struct cRay
{
    __device__ float3 pointAt(float t) const { return orig + dir * t; }

    float3 orig;
    float3 dir;
    bool done = false;
};

struct cSphere
{
    float3 center;
    float radius;
    float _not_used;
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
