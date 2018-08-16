#pragma once

#include <helper_math.h>
#include <assert.h>

#include "../Source/Maths.h"
#include "device_launch_parameters.h"

void deviceInitData(const Sphere* spheres, const Material* materials, const int spheresCount, const int numRays);

void deviceStartFrame(const Ray* rays, const uint frame);
void deviceRenderFrame(const float tMin, const float tMax, const uint depth);
void deviceEndFrame(Sample* samples);

void deviceFreeData();
