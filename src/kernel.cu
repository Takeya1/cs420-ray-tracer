// kernel.cu - GPU Ray Tracing Kernel for Hybrid Renderer
// CS420 Ray Tracer Project - Week 3

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>
#include "gpu_types.cuh"

// =========================================================
// Device Constants
// =========================================================
#define EPSILON 1e-6f
#define PI 3.14159265358979323846f

// =========================================================
// Device Vector Operations (inline for performance)
// =========================================================
struct Vec3f {
    float x, y, z;
    
    __device__ Vec3f() : x(0), y(0), z(0) {}
    __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __device__ Vec3f operator+(const Vec3f& v) const {
        return Vec3f(x + v.x, y + v.y, z + v.z);
    }
    
    __device__ Vec3f operator-(const Vec3f& v) const {
        return Vec3f(x - v.x, y - v.y, z - v.z);
    }
    
    __device__ Vec3f operator*(float t) const {
        return Vec3f(x * t, y * t, z * t);
    }
    
    __device__ Vec3f operator*(const Vec3f& v) const {
        return Vec3f(x * v.x, y * v.y, z * v.z);
    }
    
    __device__ float dot(const Vec3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }
    
    __device__ float length() const {
        return sqrtf(length_squared());
    }
    
    __device__ Vec3f normalized() const {
        float len = length();
        if (len > EPSILON) {
            return Vec3f(x / len, y / len, z / len);
        }
        return *this;
    }
};

// =========================================================
// Device Ray Structure
// =========================================================
struct Rayf {
    Vec3f origin;
    Vec3f direction;
    
    __device__ Rayf() {}
    __device__ Rayf(const Vec3f& o, const Vec3f& d) : origin(o), direction(d) {}
    
    __device__ Vec3f at(float t) const {
        return origin + direction * t;
    }
};

// =========================================================
// Sphere Intersection (from packed float array)
// =========================================================
// Sphere layout: center(3) + radius(1) + color(3) + reflectivity(1) = 8 floats
__device__ bool intersect_sphere(const Rayf& ray, const float* sphere_data,
                                  float t_min, float t_max, float& t_hit) {
    Vec3f center(sphere_data[0], sphere_data[1], sphere_data[2]);
    float radius = sphere_data[3];
    
    Vec3f oc = ray.origin - center;
    
    float a = ray.direction.dot(ray.direction);
    float half_b = oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0) return false;
    
    float sqrt_d = sqrtf(discriminant);
    
    // Find nearest root in acceptable range
    float root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }
    
    t_hit = root;
    return true;
}

// =========================================================
// Find Closest Intersection
// =========================================================
__device__ bool find_intersection(const Rayf& ray, const float* spheres, int num_spheres,
                                   float& t_hit, int& sphere_idx) {
    t_hit = FLT_MAX;
    sphere_idx = -1;
    
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray, &spheres[i * 8], EPSILON, t_hit, t)) {
            t_hit = t;
            sphere_idx = i;
        }
    }
    
    return sphere_idx >= 0;
}

// =========================================================
// Compute Shading
// =========================================================
// Light layout: position(3) + color(3) + intensity(1) = 7 floats
__device__ Vec3f compute_shading(const Vec3f& hit_point, const Vec3f& normal,
                                  const Vec3f& view_dir, const Vec3f& material_color,
                                  const float* spheres, int num_spheres,
                                  const float* lights, int num_lights,
                                  const Vec3f& ambient) {
    Vec3f color = material_color * ambient;
    
    for (int i = 0; i < num_lights; i++) {
        const float* light = &lights[i * 7];
        Vec3f light_pos(light[0], light[1], light[2]);
        Vec3f light_color(light[3], light[4], light[5]);
        float light_intensity = light[6];
        
        // Direction to light
        Vec3f light_dir = (light_pos - hit_point).normalized();
        float light_dist = (light_pos - hit_point).length();
        
        // Shadow ray
        Rayf shadow_ray(hit_point + normal * EPSILON, light_dir);
        float shadow_t;
        int shadow_idx;
        bool in_shadow = find_intersection(shadow_ray, spheres, num_spheres,
                                           shadow_t, shadow_idx);
        
        if (in_shadow && shadow_t < light_dist) {
            continue;  // Point is in shadow
        }
        
        // Diffuse (Lambertian)
        float diff = fmaxf(0.0f, normal.dot(light_dir));
        Vec3f diffuse = material_color * light_color * diff * light_intensity;
        
        // Specular (Blinn-Phong)
        Vec3f half_vec = (light_dir + view_dir).normalized();
        float spec = powf(fmaxf(0.0f, normal.dot(half_vec)), 32.0f);
        Vec3f specular = light_color * spec * light_intensity * 0.5f;
        
        color = color + diffuse + specular;
    }
    
    return color;
}

// =========================================================
// Sky/Background Color
// =========================================================
__device__ Vec3f sky_color(const Rayf& ray) {
    float t = 0.5f * (ray.direction.normalized().y + 1.0f);
    return Vec3f(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3f(0.5f, 0.7f, 1.0f) * t;
}

// =========================================================
// Main Ray Tracing Function
// =========================================================
__device__ Vec3f trace_ray(Rayf ray, const float* spheres, int num_spheres,
                           const float* lights, int num_lights,
                           int max_depth) {
    Vec3f accumulated_color(0, 0, 0);
    Vec3f attenuation(1, 1, 1);
    Vec3f ambient(0.1f, 0.1f, 0.1f);
    
    for (int depth = 0; depth < max_depth; depth++) {
        float t_hit;
        int sphere_idx;
        
        if (!find_intersection(ray, spheres, num_spheres, t_hit, sphere_idx)) {
            // No hit - add sky color and terminate
            accumulated_color = accumulated_color + sky_color(ray) * attenuation;
            break;
        }
        
        // Get sphere data
        const float* sphere = &spheres[sphere_idx * 8];
        Vec3f center(sphere[0], sphere[1], sphere[2]);
        //float radius = sphere[3];
        Vec3f material_color(sphere[4], sphere[5], sphere[6]);
        float reflectivity = sphere[7];
        
        // Calculate hit point and normal
        Vec3f hit_point = ray.at(t_hit);
        Vec3f normal = (hit_point - center).normalized();
        Vec3f view_dir = (ray.origin - hit_point).normalized();
        
        // Compute shading
        Vec3f shaded_color = compute_shading(hit_point, normal, view_dir,
                                              material_color, spheres, num_spheres,
                                              lights, num_lights, ambient);
        
        // Add contribution (weighted by current attenuation and non-reflective portion)
        accumulated_color = accumulated_color + shaded_color * attenuation * (1.0f - reflectivity);
        
        // If not reflective or last bounce, stop
        if (reflectivity < EPSILON || depth == max_depth - 1) {
            break;
        }
        
        // Setup reflection ray for next iteration
        Vec3f incident = ray.direction.normalized();
        float cos_i = -incident.dot(normal);
        Vec3f reflect_dir = incident + normal * (2.0f * cos_i);
        
        ray = Rayf(hit_point + normal * EPSILON, reflect_dir.normalized());
        attenuation = attenuation * reflectivity;
    }
    
    return accumulated_color;
}

// =========================================================
// Camera Ray Generation
// =========================================================
// Camera params layout:
// [0-2]: origin (lookfrom)
// [3-5]: lower_left_corner
// [6-8]: horizontal
// [9-11]: vertical
__device__ Rayf get_camera_ray(const float* camera_params, int px, int py,
                                int image_width, int image_height) {
    Vec3f origin(camera_params[0], camera_params[1], camera_params[2]);
    Vec3f lower_left(camera_params[3], camera_params[4], camera_params[5]);
    Vec3f horizontal(camera_params[6], camera_params[7], camera_params[8]);
    Vec3f vertical(camera_params[9], camera_params[10], camera_params[11]);
    
    float u = (float(px) + 0.5f) / float(image_width);
    float v = (float(py) + 0.5f) / float(image_height);
    
    Vec3f direction = lower_left + horizontal * u + vertical * v - origin;
    
    return Rayf(origin, direction.normalized());
}

// =========================================================
// Main Rendering Kernel
// =========================================================
__global__ void render_tile_kernel(
    float* framebuffer,
    const float* spheres, int num_spheres,
    const float* lights, int num_lights,
    const float* camera_params,
    int tile_x, int tile_y,
    int tile_width, int tile_height,
    int image_width, int image_height,
    int max_depth
) {
    // Calculate pixel coordinates within tile
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (local_x >= tile_width || local_y >= tile_height) return;
    
    // Global pixel coordinates
    int px = tile_x + local_x;
    int py = tile_y + local_y;
    
    // Bounds check for image
    if (px >= image_width || py >= image_height) return;
    
    // Generate ray and trace
    Rayf ray = get_camera_ray(camera_params, px, py, image_width, image_height);
    Vec3f color = trace_ray(ray, spheres, num_spheres, lights, num_lights, max_depth);
    
    // Clamp color
    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    // Write to framebuffer (global coordinates)
    int fb_idx = (py * image_width + px) * 3;
    framebuffer[fb_idx + 0] = color.x;
    framebuffer[fb_idx + 1] = color.y;
    framebuffer[fb_idx + 2] = color.z;
}

// =========================================================
// Kernel Launch Function (called from C++)
// =========================================================
extern "C" void launch_gpu_kernel(
    float* d_framebuffer,
    float* d_spheres, int num_spheres,
    float* d_lights, int num_lights,
    float* d_camera_params,
    int tile_x, int tile_y,
    int tile_width, int tile_height,
    int image_width, int image_height,
    int max_depth,
    cudaStream_t stream
) {
    // Configure block and grid dimensions
    // Use 16x16 thread blocks (256 threads, good occupancy)
    dim3 block_size(16, 16);
    dim3 grid_size(
        (tile_width + block_size.x - 1) / block_size.x,
        (tile_height + block_size.y - 1) / block_size.y
    );
    
    // Launch kernel
    render_tile_kernel<<<grid_size, block_size, 0, stream>>>(
        d_framebuffer,
        d_spheres, num_spheres,
        d_lights, num_lights,
        d_camera_params,
        tile_x, tile_y,
        tile_width, tile_height,
        image_width, image_height,
        max_depth
    );
}

// =========================================================
// Additional Utility: Full Image Kernel (for comparison)
// =========================================================
__global__ void render_full_image_kernel(
    float* framebuffer,
    const float* spheres, int num_spheres,
    const float* lights, int num_lights,
    const float* camera_params,
    int image_width, int image_height,
    int max_depth
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= image_width || py >= image_height) return;
    
    Rayf ray = get_camera_ray(camera_params, px, py, image_width, image_height);
    Vec3f color = trace_ray(ray, spheres, num_spheres, lights, num_lights, max_depth);
    
    // Clamp
    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    int fb_idx = (py * image_width + px) * 3;
    framebuffer[fb_idx + 0] = color.x;
    framebuffer[fb_idx + 1] = color.y;
    framebuffer[fb_idx + 2] = color.z;
}

extern "C" void launch_full_render_kernel(
    float* d_framebuffer,
    float* d_spheres, int num_spheres,
    float* d_lights, int num_lights,
    float* d_camera_params,
    int image_width, int image_height,
    int max_depth
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (image_width + block_size.x - 1) / block_size.x,
        (image_height + block_size.y - 1) / block_size.y
    );
    
    render_full_image_kernel<<<grid_size, block_size>>>(
        d_framebuffer,
        d_spheres, num_spheres,
        d_lights, num_lights,
        d_camera_params,
        image_width, image_height,
        max_depth
    );
    
    cudaDeviceSynchronize();
}
// =========================================================
// Tile-Based Kernel for Hybrid Rendering
// =========================================================

__global__ void render_tile_kernel(float3* framebuffer,
                                   GPUSphere* spheres, int num_spheres,
                                   GPULight* lights, int num_lights,
                                   GPUCamera camera,
                                   int tile_x, int tile_y,
                                   int tile_width, int tile_height,
                                   int image_width, int image_height,
                                   int max_bounces) {
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (local_x >= tile_width || local_y >= tile_height) return;
    
    int x = tile_x + local_x;
    int y = tile_y + local_y;
    
    if (x >= image_width || y >= image_height) return;
    
    float u = float(x) / float(image_width);
    float v = float(y) / float(image_height);
    GPURay ray = camera.get_ray(u, v);
    
    float3 sample_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
    
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        float t_closest = 1e20f;
        int hit_idx = -1;
        
        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (spheres[i].intersect(ray, 0.001f, t_closest, t)) {
                t_closest = t;
                hit_idx = i;
            }
        }
        
        if (hit_idx < 0) {
            float3 unit_direction = float3_ops::normalize(ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 background = float3_ops::lerp(make_float3(1.0f, 1.0f, 1.0f),
                                                 make_float3(0.5f, 0.7f, 1.0f), t);
            sample_color = float3_ops::add(sample_color,
                                           float3_ops::mul_componentwise(background, attenuation));
            break;
        }
        
        float3 hit_point = ray.at(t_closest);
        float3 normal = spheres[hit_idx].normal_at(hit_point);
        float3 view_dir = float3_ops::mul(ray.direction, -1.0f);
        
        float3 ambient = float3_ops::mul(spheres[hit_idx].material.albedo, 0.1f);
        float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
        float3 specular = make_float3(0.0f, 0.0f, 0.0f);
        
        for (int l = 0; l < num_lights; l++) {
            float3 light_dir = float3_ops::normalize(
                float3_ops::sub(lights[l].position, hit_point));
            
            float diff = fmaxf(float3_ops::dot(normal, light_dir), 0.0f);
            diffuse = float3_ops::add(diffuse,
                float3_ops::mul(
                    float3_ops::mul_componentwise(spheres[hit_idx].material.albedo, lights[l].color),
                    diff * lights[l].intensity));
            
            float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
            float spec = powf(fmaxf(float3_ops::dot(view_dir, reflect_dir), 0.0f),
                              spheres[hit_idx].material.shininess);
            specular = float3_ops::add(specular,
                float3_ops::mul(lights[l].color, spec * lights[l].intensity));
        }
        
        float3 local_color = float3_ops::add(ambient, float3_ops::add(diffuse, specular));
        sample_color = float3_ops::add(sample_color,
                                       float3_ops::mul_componentwise(local_color, attenuation));
        
        ray.origin = hit_point;
        ray.direction = float3_ops::reflect(ray.direction, normal);
        attenuation = float3_ops::mul_componentwise(attenuation, spheres[hit_idx].material.albedo);
    }
    
    int pixel_idx = y * image_width + x;
    framebuffer[pixel_idx] = sample_color;
}

// =========================================================
// Wrapper callable from C++ (main_hybrid.cpp)
// =========================================================
extern "C" void launch_tile_kernel(
    float3* d_framebuffer,
    GPUSphere* d_spheres, int num_spheres,
    GPULight* d_lights, int num_lights,
    GPUCamera camera,
    int tile_x, int tile_y,
    int tile_width, int tile_height,
    int image_width, int image_height,
    int max_bounces,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((tile_width + threads.x - 1) / threads.x,
                (tile_height + threads.y - 1) / threads.y);
    
    render_tile_kernel<<<blocks, threads, 0, stream>>>(
        d_framebuffer,
        d_spheres, num_spheres,
        d_lights, num_lights,
        camera,
        tile_x, tile_y,
        tile_width, tile_height,
        image_width, image_height,
        max_bounces
    );
}