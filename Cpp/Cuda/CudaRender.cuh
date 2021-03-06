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
    __device__ float3 pointAt(float t) const { return orig + dir * t; }

    float3 orig;
    float3 dir;
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

void HitWorldDevice(const Ray* rays, const int numRays, float tMin, float tMax, Hit* hits, DeviceData data);

void freeDeviceData(const DeviceData& data);
