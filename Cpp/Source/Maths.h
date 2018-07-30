#pragma once

#include <math.h>
#include <assert.h>
#include <stdint.h>

#define kPI 3.1415926f

struct f3
{
    f3() : x(0), y(0), z(0) {}
    f3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    float sqLength() const { return x*x+y*y+z*z; }
    float length() const { return sqrtf(x*x+y*y+z*z); }
    void normalize() { float k = 1.0f / length(); x *= k; y *= k; z *= k; }
    
    f3 operator-() const { return f3(-x, -y, -z); }
    f3& operator+=(const f3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    f3& operator-=(const f3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    f3& operator*=(const f3& o) { x*=o.x; y*=o.y; z*=o.z; return *this; }
    f3& operator*=(float o) { x*=o; y*=o; z*=o; return *this; }

    float x, y, z;
};

inline void AssertUnit(const f3& v)
{
    assert(fabsf(v.sqLength() - 1.0f) < 0.01f);
}

inline f3 operator+(const f3& a, const f3& b) { return f3(a.x+b.x,a.y+b.y,a.z+b.z); }
inline f3 operator-(const f3& a, const f3& b) { return f3(a.x-b.x,a.y-b.y,a.z-b.z); }
inline f3 operator*(const f3& a, const f3& b) { return f3(a.x*b.x,a.y*b.y,a.z*b.z); }
inline f3 operator*(const f3& a, float b) { return f3(a.x*b,a.y*b,a.z*b); }
inline f3 operator*(float a, const f3& b) { return f3(a*b.x,a*b.y,a*b.z); }
inline float dot(const f3& a, const f3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline f3 cross(const f3& a, const f3& b)
{
    return f3(
                  a.y*b.z - a.z*b.y,
                  -(a.x*b.z - a.z*b.x),
                  a.x*b.y - a.y*b.x
                  );
}
inline f3 normalize(const f3& v) { float k = 1.0f / v.length(); return f3(v.x*k, v.y*k, v.z*k); }
inline f3 reflect(const f3& v, const f3& n)
{
    return v - 2*dot(v,n)*n;
}
inline bool refract(const f3& v, const f3& n, float nint, f3& outRefracted)
{
    AssertUnit(v);
    float dt = dot(v, n);
    float discr = 1.0f - nint*nint*(1-dt*dt);
    if (discr > 0)
    {
        outRefracted = nint * (v - n*dt) - n*sqrtf(discr);
        return true;
    }
    return false;
}
inline float schlick(float cosine, float ri)
{
    float r0 = (1-ri) / (1+ri);
    r0 = r0*r0;
    return r0 + (1-r0)*powf(1-cosine, 5);
}

struct Ray
{
    Ray() {}
    Ray(const f3& orig_, const f3& dir_) : orig(orig_), dir(dir_) { AssertUnit(dir); }

    f3 pointAt(float t) const { return orig + dir * t; }
    
    f3 orig;
    f3 dir;
};


struct Hit
{
    Hit() {}
    Hit(float _t, int _id) :t(_t), id(_id) {}
    float t;
    int id = -1;
};

struct Sample
{
    f3 color;
    f3 attenuation;

    Sample() : color(0, 0, 0), attenuation(1, 1, 1) {}
};


struct Sphere
{
    Sphere() : radius(1.0f), invRadius(0.0f) {}
    Sphere(f3 center_, float radius_) : center(center_), radius(radius_), invRadius(0.0f) {}
    
    void UpdateDerivedData() { invRadius = 1.0f/radius; }
    f3 normalAt(const f3& pos) const { return (pos - center) * invRadius; }
    
    f3 center;
    float radius;
    float invRadius;
};


bool HitSphere(const Ray& r, const Sphere& s, float tMin, float tMax, float& outHitT);

float RandomFloat01(uint32_t& state);
f3 RandomInUnitDisk(uint32_t& state);
f3 RandomInUnitSphere(uint32_t& state);
f3 RandomUnitVector(uint32_t& state);

struct Camera
{
    Camera() {}
    // vfov is top to bottom in degrees
    Camera(const f3& lookFrom, const f3& lookAt, const f3& vup, float vfov, float aspect, float aperture, float focusDist)
    {
        lensRadius = aperture / 2;
        float theta = vfov*kPI/180;
        float halfHeight = tanf(theta/2);
        float halfWidth = aspect * halfHeight;
        origin = lookFrom;
        w = normalize(lookFrom - lookAt);
        u = normalize(cross(vup, w));
        v = cross(w, u);
        lowerLeftCorner = origin - halfWidth*focusDist*u - halfHeight*focusDist*v - focusDist*w;
        horizontal = 2*halfWidth*focusDist*u;
        vertical = 2*halfHeight*focusDist*v;
    }
    
    Ray GetRay(float s, float t, uint32_t& state) const
    {
        f3 rd = lensRadius * RandomInUnitDisk(state);
        f3 offset = u * rd.x + v * rd.y;
        return Ray(origin + offset, normalize(lowerLeftCorner + s*horizontal + t*vertical - origin - offset));
    }
    
    f3 origin;
    f3 lowerLeftCorner;
    f3 horizontal;
    f3 vertical;
    f3 u, v, w;
    float lensRadius;
};

