#ifndef GPU_TYPES_CUH
#define GPU_TYPES_CUH

#include <cuda_runtime.h>

// =========================================================
// GPU Vector Operations
// =========================================================
struct float3_ops {
    __host__ __device__ static float3 make(float x, float y, float z) {
        return make_float3(x, y, z);
    }

    __host__ __device__ static float3 add(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __host__ __device__ static float3 sub(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __host__ __device__ static float3 mul(const float3& a, float t) {
        return make_float3(a.x * t, a.y * t, a.z * t);
    }

    __host__ __device__ static float3 mul_componentwise(const float3& a, const float3& b) {
        return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    __host__ __device__ static float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ static float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __host__ __device__ static float length(const float3& v) {
        return sqrtf(dot(v, v));
    }

    __host__ __device__ static float3 normalize(const float3& v) {
        float len = length(v);
        return make_float3(v.x/len, v.y/len, v.z/len);
    }

    __host__ __device__ static float3 reflect(const float3& v, const float3& n) {
        return sub(v, mul(n, 2.0f * dot(v, n)));
    }

    __host__ __device__ static float3 lerp(const float3& a, const float3& b, float t) {
        return add(mul(a, 1.0f - t), mul(b, t));
    }
};

// =========================================================
// GPU Ray
// =========================================================
struct GPURay {
    float3 origin;
    float3 direction;
    
    __device__ float3 at(float t) const {
        return float3_ops::add(origin, float3_ops::mul(direction, t));
    }
};

// =========================================================
// GPU Material, Sphere, Light, Camera
// =========================================================
struct GPUMaterial {
    float3 albedo;
    float metallic;
    float roughness;
    float shininess;
};

struct GPUSphere {
    float3 center;
    float radius;
    GPUMaterial material;
    
    __device__ bool intersect(const GPURay& ray, float t_min, float t_max, float& t) const {
        float3 oc = float3_ops::sub(ray.origin, center);
        float a = float3_ops::dot(ray.direction, ray.direction);
        float b = float3_ops::dot(oc, ray.direction);
        float c = float3_ops::dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        
        if (discriminant < 0) return false;
        
        float sqrt_disc = sqrtf(discriminant);
        float root = (-b - sqrt_disc) / a;
        if (root < t_max && root > t_min) {
            t = root;
            return true;
        }
        root = (-b + sqrt_disc) / a;
        if (root < t_max && root > t_min) {
            t = root;
            return true;
        }
        return false;
    }
    
    __device__ float3 normal_at(const float3& point) const {
        return float3_ops::normalize(float3_ops::sub(point, center));
    }
};

struct GPULight {
    float3 position;
    float3 color;
    float intensity;
};

struct GPUCamera {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
    
    __device__ GPURay get_ray(float u, float v) const {
        float3 direction = float3_ops::add(
            lower_left,
            float3_ops::add(
                float3_ops::mul(horizontal, u),
                float3_ops::mul(vertical, v)
            )
        );
        direction = float3_ops::sub(direction, origin);
        
        GPURay ray;
        ray.origin = origin;
        ray.direction = float3_ops::normalize(direction);
        return ray;
    }
};

#endif // GPU_TYPES_CUH