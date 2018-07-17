#include "Config.h"
#include "Test.h"
#include "Maths.h"
#include <algorithm>
#include <atomic>

static Sphere s_Spheres[] =
{
    {float3(0,-100.5,-1), 100},
    {float3(2,0,-1), 0.5f},
    {float3(0,0,-1), 0.5f},
    {float3(-2,0,-1), 0.5f},
    {float3(2,0,1), 0.5f},
    {float3(0,0,1), 0.5f},
    {float3(-2,0,1), 0.5f},
    {float3(0.5f,1,0.5f), 0.5f},
    {float3(-1.5f,1.5f,0.f), 0.3f},
};
const int kSphereCount = sizeof(s_Spheres) / sizeof(s_Spheres[0]);

struct Material
{
    enum Type { Lambert, Metal, Dielectric };
    Type type;
    float3 albedo;
    float3 emissive;
    float roughness;
    float ri;
};

static Material s_SphereMats[kSphereCount] =
{
    { Material::Lambert, float3(0.8f, 0.8f, 0.8f), float3(0,0,0), 0, 0, },
    { Material::Lambert, float3(0.8f, 0.4f, 0.4f), float3(0,0,0), 0, 0, },
    { Material::Lambert, float3(0.4f, 0.8f, 0.4f), float3(0,0,0), 0, 0, },
    { Material::Metal, float3(0.4f, 0.4f, 0.8f), float3(0,0,0), 0, 0 },
    { Material::Metal, float3(0.4f, 0.8f, 0.4f), float3(0,0,0), 0, 0 },
    { Material::Metal, float3(0.4f, 0.8f, 0.4f), float3(0,0,0), 0.2f, 0 },
    { Material::Metal, float3(0.4f, 0.8f, 0.4f), float3(0,0,0), 0.6f, 0 },
    { Material::Dielectric, float3(0.4f, 0.4f, 0.4f), float3(0,0,0), 0, 1.5f },
    { Material::Lambert, float3(0.8f, 0.6f, 0.2f), float3(30,25,15), 0, 0 },
};

static Camera s_Cam;

const float kMinT = 0.001f;
const float kMaxT = 1.0e7f;
const int kMaxDepth = 10;


void HitWorld(const Ray* rays, const int num_rays, float tMin, float tMax, Hit* hits)
{
    for (int rIdx = 0; rIdx < num_rays; rIdx++)
    {
        const Ray& r = rays[rIdx];
        if (r.done)
            continue;

        Hit tmpHit;
        float closest = tMax;
        Hit outHit;
        outHit.t = -1;
        for (int i = 0; i < kSphereCount; ++i)
        {
            if (HitSphere(r, s_Spheres[i], tMin, closest, tmpHit))
            {
                closest = tmpHit.t;
                outHit = tmpHit;
                outHit.id = i;
            }
        }

        hits[rIdx] = outHit;
    }
}


static bool Scatter(const Material& mat, const Ray& r_in, const Hit& rec, float3& attenuation, Ray& scattered, float3& outLightE, int& inoutRayCount, uint32_t& state)
{
    outLightE = float3(0,0,0);
    if (mat.type == Material::Lambert)
    {
        // random point on unit sphere that is tangent to the hit point
        float3 target = rec.pos + rec.normal + RandomUnitVector(state);
        scattered = Ray(rec.pos, normalize(target - rec.pos));
        attenuation = mat.albedo;
        
        // sample lights
#if DO_LIGHT_SAMPLING
        for (int i = 0; i < kSphereCount; ++i)
        {
            const Material& smat = s_SphereMats[i];
            if (smat.emissive.x <= 0 && smat.emissive.y <= 0 && smat.emissive.z <= 0)
                continue; // skip non-emissive
            if (&mat == &smat)
                continue; // skip self
            const Sphere& s = s_Spheres[i];
            
            // create a random direction towards sphere
            // coord system for sampling: sw, su, sv
            float3 sw = normalize(s.center - rec.pos);
            float3 su = normalize(cross(fabs(sw.x)>0.01f ? float3(0,1,0):float3(1,0,0), sw));
            float3 sv = cross(sw, su);
            // sample sphere by solid angle
            float cosAMax = sqrtf(1.0f - s.radius*s.radius / (rec.pos-s.center).sqLength());
            float eps1 = RandomFloat01(state), eps2 = RandomFloat01(state);
            float cosA = 1.0f - eps1 + eps1 * cosAMax;
            float sinA = sqrtf(1.0f - cosA*cosA);
            float phi = 2 * kPI * eps2;
            float3 l = su * cosf(phi) * sinA + sv * sin(phi) * sinA + sw * cosA;
            l.normalize();
            
            // shoot shadow ray
            Hit lightHit;
            int hitID;
            ++inoutRayCount;
            if (HitWorld(Ray(rec.pos, l), kMinT, kMaxT, lightHit, hitID) && hitID == i)
            {
                float omega = 2 * kPI * (1-cosAMax);
                
                float3 rdir = r_in.dir;
                AssertUnit(rdir);
                float3 nl = dot(rec.normal, rdir) < 0 ? rec.normal : -rec.normal;
                outLightE += (mat.albedo * smat.emissive) * (std::max(0.0f, dot(l, nl)) * omega / kPI);
            }
        }
#endif
        return true;
    }
    else if (mat.type == Material::Metal)
    {
        AssertUnit(r_in.dir); AssertUnit(rec.normal);
        float3 refl = reflect(r_in.dir, rec.normal);
        // reflected ray, and random inside of sphere based on roughness
        float roughness = mat.roughness;
#if DO_MITSUBA_COMPARE
        roughness = 0; // until we get better BRDF for metals
#endif
        scattered = Ray(rec.pos, normalize(refl + roughness*RandomInUnitSphere(state)));
        attenuation = mat.albedo;
        return dot(scattered.dir, rec.normal) > 0;
    }
    else if (mat.type == Material::Dielectric)
    {
        AssertUnit(r_in.dir); AssertUnit(rec.normal);
        float3 outwardN;
        float3 rdir = r_in.dir;
        float3 refl = reflect(rdir, rec.normal);
        float nint;
        attenuation = float3(1,1,1);
        float3 refr;
        float reflProb;
        float cosine;
        if (dot(rdir, rec.normal) > 0)
        {
            outwardN = -rec.normal;
            nint = mat.ri;
            cosine = mat.ri * dot(rdir, rec.normal);
        }
        else
        {
            outwardN = rec.normal;
            nint = 1.0f / mat.ri;
            cosine = -dot(rdir, rec.normal);
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
            scattered = Ray(rec.pos, normalize(refl));
        else
            scattered = Ray(rec.pos, normalize(refr));
    }
    else
    {
        attenuation = float3(1,0,1);
        return false;
    }
    return true;
}


static bool ScatterNoLightSampling(const Material& mat, const Ray& r_in, const Hit& rec, float3& attenuation, Ray& scattered, uint32_t& state)
{
    if (mat.type == Material::Lambert)
    {
        // random point on unit sphere that is tangent to the hit point
        float3 target = rec.pos + rec.normal + RandomUnitVector(state);
        scattered = Ray(rec.pos, normalize(target - rec.pos));
        attenuation = mat.albedo;

        return true;
    }
    else if (mat.type == Material::Metal)
    {
        AssertUnit(r_in.dir); AssertUnit(rec.normal);
        float3 refl = reflect(r_in.dir, rec.normal);
        // reflected ray, and random inside of sphere based on roughness
        float roughness = mat.roughness;
#if DO_MITSUBA_COMPARE
        roughness = 0; // until we get better BRDF for metals
#endif
        scattered = Ray(rec.pos, normalize(refl + roughness * RandomInUnitSphere(state)));
        attenuation = mat.albedo;
        return dot(scattered.dir, rec.normal) > 0;
    }
    else if (mat.type == Material::Dielectric)
    {
        AssertUnit(r_in.dir); AssertUnit(rec.normal);
        float3 outwardN;
        float3 rdir = r_in.dir;
        float3 refl = reflect(rdir, rec.normal);
        float nint;
        attenuation = float3(1, 1, 1);
        float3 refr;
        float reflProb;
        float cosine;
        if (dot(rdir, rec.normal) > 0)
        {
            outwardN = -rec.normal;
            nint = mat.ri;
            cosine = mat.ri * dot(rdir, rec.normal);
        }
        else
        {
            outwardN = rec.normal;
            nint = 1.0f / mat.ri;
            cosine = -dot(rdir, rec.normal);
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
            scattered = Ray(rec.pos, normalize(refl));
        else
            scattered = Ray(rec.pos, normalize(refr));
    }
    else
    {
        attenuation = float3(1, 0, 1);
        return false;
    }
    return true;
}

static void TraceIterative(Ray* rays, Sample* samples, const int num_rays, int& inoutRayCount, uint32_t& state)
{
    Hit* hits = new Hit[num_rays];

    for (int depth = 0; depth <= kMaxDepth; depth++)
    {
        HitWorld(rays, num_rays, kMinT, kMaxT, hits);
        for (int rIdx = 0; rIdx < num_rays; rIdx++)
        {
            const Ray& r = rays[rIdx];
            if (r.done)
                continue;

            Hit& rec = hits[rIdx];
            Sample& sample = samples[rIdx];

            ++inoutRayCount;
            if (rec.t > 0)
            {
                Ray scattered;
                const Material& mat = s_SphereMats[rec.id];
                float3 local_attenuation;
                sample.color += mat.emissive * sample.attenuation;
                if (depth < kMaxDepth && ScatterNoLightSampling(mat, r, rec, local_attenuation, scattered, state))
                {
                    sample.attenuation *= local_attenuation;
                    rays[rIdx] = scattered;
                }
                else
                {
                    rays[rIdx].done = true;
                }
            }
            else
            {
                // sky
#if DO_MITSUBA_COMPARE
                sample.color += sample.attenuation * float3(0.15f, 0.21f, 0.3f); // easier compare with Mitsuba's constant environment light
#else
                float3 unitDir = r.dir;
                float t = 0.5f*(unitDir.y + 1.0f);
                sample.color += sample.attenuation * ((1.0f - t)*float3(1.0f, 1.0f, 1.0f) + t * float3(0.5f, 0.7f, 1.0f)) * 0.3f;
                rays[rIdx].done = true;
#endif
            }
        }
    }

    delete[] hits;
}

struct JobData
{
    float time;
    int frameCount;
    int screenWidth, screenHeight;
    float* backbuffer;
    Camera* cam;
    std::atomic<int> rayCount;
};

static void TracePixels(void* data_)
{
    JobData& data = *(JobData*)data_;
    float* backbuffer = data.backbuffer;
    float invWidth = 1.0f / data.screenWidth;
    float invHeight = 1.0f / data.screenHeight;
    float lerpFac = float(data.frameCount) / float(data.frameCount + 1);
#if !DO_PROGRESSIVE
    lerpFac = 0;
#endif
    int rayCount = 0;
    uint32_t state = (data.frameCount * 26699) | 1;

    const int num_rays = data.screenWidth*data.screenHeight*DO_SAMPLES_PER_PIXEL;

    // let's allocate a few arrays needed by the renderer
    Ray* rays = new Ray[num_rays];
    Sample* samples = new Sample[num_rays];

    // generate camera rays for all samples
    for (int y = 0, rIdx = 0; y < data.screenHeight; y++)
    {
        for (int x = 0; x < data.screenWidth; x++)
        {
            for (int s = 0; s < DO_SAMPLES_PER_PIXEL; s++, ++rIdx)
            {
                float u = float(x + RandomFloat01(state)) * invWidth;
                float v = float(y + RandomFloat01(state)) * invHeight;
                rays[rIdx] = data.cam->GetRay(u, v, state);
            }
        }
    }

    // trace all samples through the scene
    TraceIterative(rays, samples, num_rays, rayCount, state);

    // compute cumulated color for all samples
    for (int y = 0, rIdx = 0; y < data.screenHeight; y++)
    {
        for (int x = 0; x < data.screenWidth; x++)
        {
            float3 col(0, 0, 0);
            for (int s = 0; s < DO_SAMPLES_PER_PIXEL; s++, ++rIdx)
            {
                col += samples[rIdx].color;
            }
            col *= 1.0f / float(DO_SAMPLES_PER_PIXEL);

            float3 prev(backbuffer[0], backbuffer[1], backbuffer[2]);
            col = prev * lerpFac + col * (1 - lerpFac);
            backbuffer[0] = col.x;
            backbuffer[1] = col.y;
            backbuffer[2] = col.z;
            backbuffer += 4;
        }
    }

    data.rayCount += rayCount;

    // don't forget to delete allocated arrays
    delete[] rays;
    delete[] samples;
}

void UpdateTest(float time, int frameCount, int screenWidth, int screenHeight)
{
    float3 lookfrom(0, 2, 3);
    float3 lookat(0, 0, 0);
    float distToFocus = 3;
#if DO_MITSUBA_COMPARE
    float aperture = 0.0f;
#else
    float aperture = 0.1f;
#endif

    for (int i = 0; i < kSphereCount; ++i)
        s_Spheres[i].UpdateDerivedData();

    s_Cam = Camera(lookfrom, lookat, float3(0, 1, 0), 60, float(screenWidth) / float(screenHeight), aperture, distToFocus);
}

void DrawTest(float time, int frameCount, int screenWidth, int screenHeight, float* backbuffer, int& outRayCount)
{    
    JobData args;
    args.time = time;
    args.frameCount = frameCount;
    args.screenWidth = screenWidth;
    args.screenHeight = screenHeight;
    args.backbuffer = backbuffer;
    args.cam = &s_Cam;
    args.rayCount = 0;
    //for (int y = 0; y < screenHeight; y++)
    //    for (int x = 0; x < screenWidth; x++)
    //        TracePixelJob(x, y, &args);
    TracePixels(&args);
    outRayCount = args.rayCount;
}

void GetObjectCount(int& outCount, int& outObjectSize, int& outMaterialSize, int& outCamSize)
{
    outCount = kSphereCount;
    outObjectSize = sizeof(Sphere);
    outMaterialSize = sizeof(Material);
    outCamSize = sizeof(Camera);
}

void GetSceneDesc(void* outObjects, void* outMaterials, void* outCam)
{
    memcpy(outObjects, s_Spheres, kSphereCount * sizeof(s_Spheres[0]));
    memcpy(outMaterials, s_SphereMats, kSphereCount * sizeof(s_SphereMats[0]));
    memcpy(outCam, &s_Cam, sizeof(s_Cam));
}
