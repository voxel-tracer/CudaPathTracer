#include "Config.h"
#include "Test.h"
#include "Maths.h"
#include <algorithm>
#include <atomic>

#if DO_CUDA_RENDER
#include "../Cuda/CudaRender.cuh"
#endif // DO_CUDA_RENDER


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

struct Material
{
    enum Type { Lambert, Metal, Dielectric };
    Type type;
    f3 albedo;
    f3 emissive;
    float roughness;
    float ri;
};

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
const int kMaxDepth = 10;

struct RendererData
{
    int frameCount;
    int screenWidth, screenHeight;
    float* backbuffer;
    Camera* cam;
    int numRays;
    Ray* rays;
    Hit* hits;
    Sample* samples;
#if DO_CUDA_RENDER
    DeviceData deviceData;
#endif // DO_CUDA_RENDER
};


void HitWorld(const Ray* rays, const int num_rays, float tMin, float tMax, Hit* hits)
{
    for (int rIdx = 0; rIdx < num_rays; rIdx++)
    {
        const Ray& r = rays[rIdx];
        if (r.isDone())
            continue;

        float closest = tMax, hitT;
        int hitId = -1;
        for (int i = 0; i < kSphereCount; ++i)
        {
            if (HitSphere(r, s_Spheres[i], tMin, closest, hitT))
            {
                closest = hitT;
                hitId = i;
            }
        }

        hits[rIdx] = Hit(hitT, hitId);
    }
}

static bool ScatterNoLightSampling(const Material& mat, const Ray& r_in, const Hit& rec, f3& attenuation, Ray& scattered, uint32_t& state)
{
    const f3 hitPos = r_in.pointAt(rec.t);
    const f3 hitNormal = s_Spheres[rec.id].normalAt(hitPos);

    if (mat.type == Material::Lambert)
    {
        // random point on unit sphere that is tangent to the hit point
        f3 target = hitPos + hitNormal + RandomUnitVector(state);
        scattered = Ray(hitPos, normalize(target - hitPos));
        attenuation = mat.albedo;

        return true;
    }
    else if (mat.type == Material::Metal)
    {
        AssertUnit(r_in.dir); AssertUnit(hitNormal);
        f3 refl = reflect(r_in.dir, hitNormal);
        // reflected ray, and random inside of sphere based on roughness
        float roughness = mat.roughness;
#if DO_MITSUBA_COMPARE
        roughness = 0; // until we get better BRDF for metals
#endif
        scattered = Ray(hitPos, normalize(refl + roughness * RandomInUnitSphere(state)));
        attenuation = mat.albedo;
        return dot(scattered.dir, hitNormal) > 0;
    }
    else if (mat.type == Material::Dielectric)
    {
        AssertUnit(r_in.dir); AssertUnit(hitNormal);
        f3 outwardN;
        f3 rdir = r_in.dir;
        f3 refl = reflect(rdir, hitNormal);
        float nint;
        attenuation = f3(1, 1, 1);
        f3 refr;
        float reflProb;
        float cosine;
        if (dot(rdir, hitNormal) > 0)
        {
            outwardN = -hitNormal;
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
            reflProb = schlick(cosine, mat.ri);
        }
        else
        {
            reflProb = 1;
        }
        if (RandomFloat01(state) < reflProb)
            scattered = Ray(hitPos, normalize(refl));
        else
            scattered = Ray(hitPos, normalize(refr));
    }
    else
    {
        attenuation = f3(1, 0, 1);
        return false;
    }
    return true;
}

static void TraceIterative(const RendererData& data, int& inoutRayCount, uint32_t& state)
{
    for (int rIdx = 0; rIdx < data.numRays; rIdx++)
    {
        Sample& sample = data.samples[rIdx];
        sample.color = f3(0, 0, 0);
        sample.attenuation = f3(1, 1, 1);
    }

    for (int depth = 0; depth <= kMaxDepth; depth++)
    {
#if DO_CUDA_RENDER
        HitWorldDevice(data.rays, kMinT, kMaxT, data.hits, data.deviceData);
#else
        HitWorld(data.rays, data.numRays, kMinT, kMaxT, data.hits);
#endif
        for (int rIdx = 0; rIdx < data.numRays; rIdx++)
        {
            const Ray& r = data.rays[rIdx];
            if (r.isDone())
                continue;

            const Hit& rec = data.hits[rIdx];
            Sample& sample = data.samples[rIdx];

            ++inoutRayCount;
            if (rec.id >= 0)
            {
                Ray scattered;
                const Material& mat = s_SphereMats[rec.id];
                f3 local_attenuation;
                sample.color += mat.emissive * sample.attenuation;
                if (depth < kMaxDepth && ScatterNoLightSampling(mat, r, rec, local_attenuation, scattered, state))
                {
                    sample.attenuation *= local_attenuation;
                    data.rays[rIdx] = scattered;
                }
                else
                {
                    data.rays[rIdx].setDone();
                }
            }
            else
            {
                // sky
#if DO_MITSUBA_COMPARE
                sample.color += sample.attenuation * f3(0.15f, 0.21f, 0.3f); // easier compare with Mitsuba's constant environment light
#else
                f3 unitDir = r.dir;
                float t = 0.5f*(unitDir.y + 1.0f);
                sample.color += sample.attenuation * ((1.0f - t)*f3(1.0f, 1.0f, 1.0f) + t * f3(0.5f, 0.7f, 1.0f)) * 0.3f;
                data.rays[rIdx].setDone();
#endif
            }
        }
    }
}

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
    uint32_t state = (data.frameCount * 26699) | 1;


    // generate camera rays for all samples
    for (int y = 0, rIdx = 0; y < data.screenHeight; y++)
    {
        for (int x = 0; x < data.screenWidth; x++)
        {
            for (int s = 0; s < DO_SAMPLES_PER_PIXEL; s++, ++rIdx)
            {
                float u = float(x + RandomFloat01(state)) * invWidth;
                float v = float(y + RandomFloat01(state)) * invHeight;
                data.rays[rIdx] = data.cam->GetRay(u, v, state);
            }
        }
    }

    // trace all samples through the scene
    TraceIterative(data, rayCount, state);

    // compute cumulated color for all samples
    for (int y = 0, rIdx = 0; y < data.screenHeight; y++)
    {
        for (int x = 0; x < data.screenWidth; x++)
        {
            f3 col(0, 0, 0);
            for (int s = 0; s < DO_SAMPLES_PER_PIXEL; s++, ++rIdx)
            {
                col += data.samples[rIdx].color;
            }
            col *= 1.0f / float(DO_SAMPLES_PER_PIXEL);

            f3 prev(backbuffer[0], backbuffer[1], backbuffer[2]);
            col = prev * lerpFac + col * (1 - lerpFac);
            backbuffer[0] = col.x;
            backbuffer[1] = col.y;
            backbuffer[2] = col.z;
            backbuffer += 4;
        }
    }

    return rayCount;
}

void Render(int screenWidth, int screenHeight, float* backbuffer, int& outRayCount)
{
    f3 lookfrom(0, 2, 3);
    f3 lookat(0, 0, 0);
    float distToFocus = 3;
#if DO_MITSUBA_COMPARE
    float aperture = 0.0f;
#else
    float aperture = 0.1f;
#endif

    for (int i = 0; i < kSphereCount; ++i)
        s_Spheres[i].UpdateDerivedData();

    s_Cam = Camera(lookfrom, lookat, f3(0, 1, 0), 60, float(screenWidth) / float(screenHeight), aperture, distToFocus);

    // let's allocate a few arrays needed by the renderer
    int numRays = screenWidth * screenHeight * DO_SAMPLES_PER_PIXEL;
    Ray* rays = NULL;
    Hit* hits = NULL;
#if DO_CUDA_RENDER
    cudaMallocHost((void**)&rays, numRays * sizeof(Ray));
    cudaMallocHost((void**)&hits, numRays * sizeof(Hit));
#else
    rays = new Ray[numRays];
    hits = new Hit[numRays];
#endif

    Sample* samples = new Sample[numRays];

    RendererData args;
    args.screenWidth = screenWidth;
    args.screenHeight = screenHeight;
    args.backbuffer = backbuffer;
    args.cam = &s_Cam;
    args.rays = rays;
    args.samples = samples;
    args.hits = hits;
    args.numRays = numRays;

#if DO_CUDA_RENDER
    initDeviceData(s_Spheres, kSphereCount, numRays, args.deviceData);
#endif // DO_CUDA_RENDER

    for (int frame = 0; frame < kNumFrames; frame++)
    {
        args.frameCount = frame;
        outRayCount += TracePixels(args);
    }

#if DO_CUDA_RENDER
    cudaFreeHost(rays);
    cudaFreeHost(hits);
#else
    delete[] rays;
    delete[] hits;
#endif
    delete[] samples;

#if DO_CUDA_RENDER
    freeDeviceData(args.deviceData);
#endif // DO_CUDA_RENDER

}
