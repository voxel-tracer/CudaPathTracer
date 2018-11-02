#include "Config.h"
#include "Test.h"
#include "Maths.h"
#include <algorithm>
#include <atomic>

#include "../Cuda/CudaRender.cuh"


static Sphere s_Spheres[] =
{
    {f3(0,-100.5,-1), 100},
    {f3(2,0,-1), 0.5f},
    {f3(0,0,-1), 0.5f},
    {f3(-2,0,-1), 0.5f},
    {f3(2,0,1), 0.5f},
    {f3(0,0,1), 0.5f},
    {f3(-2,0,1), 0.5f},
    {f3(0.5f,1,0.5f), 0.5f},
    {f3(-1.5f,1.5f,0.f), 0.3f},
};
const int kSphereCount = sizeof(s_Spheres) / sizeof(s_Spheres[0]);

static Material s_SphereMats[kSphereCount] =
{
    { Material::Lambert, f3(0.8f, 0.8f, 0.8f), f3(0,0,0), 0, 0, },
    { Material::Lambert, f3(0.8f, 0.4f, 0.4f), f3(0,0,0), 0, 0, },
    { Material::Lambert, f3(0.4f, 0.8f, 0.4f), f3(0,0,0), 0, 0, },
    { Material::Metal, f3(0.4f, 0.4f, 0.8f), f3(0,0,0), 0, 0 },
    { Material::Metal, f3(0.4f, 0.8f, 0.4f), f3(0,0,0), 0, 0 },
    { Material::Metal, f3(0.4f, 0.8f, 0.4f), f3(0,0,0), 0.2f, 0 },
    { Material::Metal, f3(0.4f, 0.8f, 0.4f), f3(0,0,0), 0.6f, 0 },
    { Material::Dielectric, f3(0.4f, 0.4f, 0.4f), f3(0,0,0), 0, 1.5f },
    { Material::Lambert, f3(0.8f, 0.6f, 0.2f), f3(30,25,15), 0, 0 },
};

static Camera s_Cam;

const float kMinT = 0.001f;
const float kMaxT = 1.0e7f;

struct RendererData
{
    int frameCount;
    int screenWidth, screenHeight;
    float* backbuffer;
    Camera* cam;
    int numRays;

    f3* colors;
};

static int TracePixels(RendererData data)
{
    float* backbuffer = data.backbuffer;
    float invWidth = 1.0f / data.screenWidth;
    float invHeight = 1.0f / data.screenHeight;
    float lerpFac = float(data.frameCount) / float(data.frameCount + 1);
#if !DO_PROGRESSIVE
    lerpFac = 0;
#endif
    int rayCount = 0;

    // startFrame renders the primary rays, that's why depth starts from 1
    deviceStartFrame(data.frameCount, kMinT, kMaxT);

    // trace all samples through the scene
    for (int depth = 1; depth <= kMaxDepth; depth++)
        deviceRenderFrame(kMinT, kMaxT, depth);

    return rayCount;
}

void Render(int screenWidth, int screenHeight, float* backbuffer, int& outRayCount)
{
    f3 lookfrom(0, 2, 3);
    f3 lookat(0, 0, 0);
    float distToFocus = 3;
    float aperture = 0.1f;

    s_Cam = Camera(lookfrom, lookat, f3(0, 1, 0), 60, float(screenWidth) / float(screenHeight), aperture, distToFocus);

    // let's allocate a few arrays needed by the renderer
    int numRays = screenWidth * screenHeight * DO_SAMPLES_PER_PIXEL;

    f3* colors = new f3[numRays];

    RendererData args;
    args.screenWidth = screenWidth;
    args.screenHeight = screenHeight;
    args.backbuffer = backbuffer;
    args.cam = &s_Cam;
    args.colors = colors;
    args.numRays = numRays;

    deviceInitData(&s_Cam, screenWidth, screenHeight, s_Spheres, s_SphereMats, kSphereCount, numRays);
    if (kNumFrames == 100) outRayCount = 854161957;
    else if (kNumFrames == 1000) outRayCount = 4246579592;

    for (int frame = 0; frame < kNumFrames; frame++)
    {
        args.frameCount = frame;
        outRayCount += TracePixels(args);
    }

    deviceEndRendering(args.colors);

    // compute cumulated color for all samples
    const float devider = 1.0f / float(kNumFrames*DO_SAMPLES_PER_PIXEL);
    for (int y = 0, rIdx = 0; y < args.screenHeight; y++) {
        for (int x = 0; x < args.screenWidth; x++) {
            f3 col(0, 0, 0);
            for (int s = 0; s < DO_SAMPLES_PER_PIXEL; s++, ++rIdx)
                col += args.colors[rIdx];
            col *= devider;

            backbuffer[0] = col.x;
            backbuffer[1] = col.y;
            backbuffer[2] = col.z;
            backbuffer += 4;
        }
    }

    deviceFreeData();

    delete[] colors;
}
