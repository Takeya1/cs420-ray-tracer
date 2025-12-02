// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define CUDA math constants if not already defined (for compatibility with older CUDA versions)
#ifndef CUDART_PI
#define CUDART_PI 3.1415926535897931e+0
#endif
#ifndef CUDART_THIRD
#define CUDART_THIRD 3.3333333333333333e-1
#endif
#ifndef CUDART_SQRT_HALF_HI
#define CUDART_SQRT_HALF_HI 7.0710678118654757e-1
#endif
#ifndef CUDART_SQRT_HALF_LO
#define CUDART_SQRT_HALF_LO (-1.7171281366606629e-17)
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// =========================================================
// GPU Vector and Ray Classes (simplified for CUDA)
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

    __host__ __device__ static float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
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

struct GPURay {
    float3 origin;
    float3 direction;
    
    __device__ float3 at(float t) const {
        return float3_ops::add(origin, float3_ops::mul(direction, t));
    }
};

// =========================================================
// GPU Sphere and Material Structures
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
    
    // ===== TODO: STUDENT - IMPLEMENT GPU SPHERE INTERSECTION =====
    __device__ bool intersect(const GPURay& ray, float t_min, float t_max, float& t) const {
        // Vector from ray origin to sphere center
        float3 oc = float3_ops::sub(ray.origin, center);

        // Quadratic equation coefficients: at^2 + bt + c = 0
        float a = float3_ops::dot(ray.direction, ray.direction);
        float b = 2.0f * float3_ops::dot(oc, ray.direction);
        float c = float3_ops::dot(oc, oc) - radius * radius;

        // Discriminant
        float discriminant = b * b - 4.0f * a * c;

        // No intersection if discriminant is negative
        if (discriminant < 0.0f) {
            return false;
        }

        // Calculate the nearest intersection point
        float sqrt_discriminant = sqrtf(discriminant);
        float temp = (-b - sqrt_discriminant) / (2.0f * a);

        // Check if nearest intersection is in valid range
        if (temp < t_min || temp > t_max) {
            // Try the other root
            temp = (-b + sqrt_discriminant) / (2.0f * a);
            if (temp < t_min || temp > t_max) {
                return false;
            }
        }

        t = temp;
        return true;
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

// =========================================================
// GPU Camera
// =========================================================

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

// =========================================================
// TODO: STUDENT IMPLEMENTATION - GPU Ray Tracing Kernel
// =========================================================
// Implement the main ray tracing kernel that runs on the GPU.
// Each thread handles one pixel.
// 
// Key differences from CPU version:
// - No recursion (use iterative approach for reflections)
// - Use shared memory for frequently accessed data
// - Be careful with memory access patterns
// =========================================================

__global__ void render_kernel(float3* framebuffer, 
                             GPUSphere* spheres, int num_spheres,
                             GPULight* lights, int num_lights,
                             GPUCamera camera,
                             int width, int height,
                             int max_bounces) {
    
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. Generate ray for this pixel
    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);
    GPURay ray = camera.get_ray(u, v);

    // 2. Initialize color accumulator and attenuation
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

    // 3. Iterative ray bouncing (no recursion on GPU!)
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        // a. Find closest intersection
        float closest_t = 1e10f;
        int hit_idx = -1;

        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (spheres[i].intersect(ray, 0.001f, closest_t, t)) {
                closest_t = t;
                hit_idx = i;
            }
        }

        // b. If no hit, add background color and break
        if (hit_idx < 0) {
            // Sky gradient background
            float t = 0.5f * (ray.direction.y + 1.0f);
            float3 bg = float3_ops::lerp(
                make_float3(1.0f, 1.0f, 1.0f),
                make_float3(0.5f, 0.7f, 1.0f),
                t
            );
            color.x += bg.x * attenuation.x;
            color.y += bg.y * attenuation.y;
            color.z += bg.z * attenuation.z;
            break;
        }

        // c. Calculate shading (ambient + diffuse + specular)
        float3 hit_point = ray.at(closest_t);
        float3 normal = spheres[hit_idx].normal_at(hit_point);
        GPUMaterial mat = spheres[hit_idx].material;

        // Start with ambient lighting
        float3 shading = make_float3(0.1f, 0.1f, 0.1f);

        // Add contribution from each light
        for (int l = 0; l < num_lights; l++) {
            float3 light_dir = float3_ops::normalize(
                float3_ops::sub(lights[l].position, hit_point)
            );

            // Diffuse lighting
            float diff = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
            float3 diffuse = float3_ops::mul(lights[l].color, diff * lights[l].intensity);

            // Specular lighting (Blinn-Phong)
            float3 view_dir = float3_ops::normalize(float3_ops::mul(ray.direction, -1.0f));
            float3 half_dir = float3_ops::normalize(float3_ops::add(light_dir, view_dir));
            float spec = powf(fmaxf(0.0f, float3_ops::dot(normal, half_dir)), mat.shininess);
            float3 specular = float3_ops::mul(lights[l].color, spec * lights[l].intensity * mat.metallic);

            shading = float3_ops::add(shading, float3_ops::add(diffuse, specular));
        }

        // Apply material albedo
        shading.x *= mat.albedo.x;
        shading.y *= mat.albedo.y;
        shading.z *= mat.albedo.z;

        // e. Accumulate color with attenuation
        color.x += shading.x * attenuation.x;
        color.y += shading.y * attenuation.y;
        color.z += shading.z * attenuation.z;

        // d. If reflective, setup ray for next bounce
        if (mat.metallic > 0.01f && bounce < max_bounces - 1) {
            ray.origin = hit_point;
            ray.direction = float3_ops::reflect(ray.direction, normal);
            attenuation.x *= mat.metallic;
            attenuation.y *= mat.metallic;
            attenuation.z *= mat.metallic;
        } else {
            break;
        }
    }

    // 4. Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = color;
}

// =========================================================
// TODO: STUDENT OPTIMIZATION - Shared Memory Kernel
// =========================================================
// Implement an optimized version using shared memory for spheres
// that are accessed by all threads in a block.
// =========================================================

__global__ void render_kernel_optimized(float3* framebuffer,
                                       GPUSphere* global_spheres, int num_spheres,
                                       GPULight* lights, int num_lights,
                                       GPUCamera camera,
                                       int width, int height,
                                       int max_bounces) {
    
    // TODO: STUDENT CODE HERE
    // 1. Declare shared memory for spheres
    //    extern __shared__ GPUSphere shared_spheres[];
    // 2. Cooperatively load spheres into shared memory
    // 3. __syncthreads()
    // 4. Use shared_spheres instead of global_spheres for intersection tests
    
    // For now, just call the basic kernel logic
    render_kernel(framebuffer, global_spheres, num_spheres, 
                 lights, num_lights, camera, width, height, max_bounces);
}

// =========================================================
// Host Functions
// =========================================================

void write_ppm(const std::string& filename, const std::vector<float3>& framebuffer,
               int width, int height) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            float3 color = framebuffer[j * width + i];
            int r = int(255.99f * fminf(1.0f, color.x));
            int g = int(255.99f * fminf(1.0f, color.y));
            int b = int(255.99f * fminf(1.0f, color.z));
            file << r << " " << g << " " << b << "\n";
        }
    }
}

GPUCamera setup_camera(int width, int height) {
    // Camera parameters
    float3 lookfrom = make_float3(0, 2, 5);
    float3 lookat = make_float3(0, 0, -20);
    float3 vup = make_float3(0, 1, 0);
    float vfov = 60.0f;
    float aspect = float(width) / float(height);
    
    // Calculate camera basis
    float theta = vfov * M_PI / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect * viewport_height;
    float focal_length = 1.0f;
    
    float3 w = float3_ops::normalize(float3_ops::sub(lookfrom, lookat));
    float3 u = float3_ops::normalize(make_float3(
        vup.y * w.z - vup.z * w.y,
        vup.z * w.x - vup.x * w.z,
        vup.x * w.y - vup.y * w.x
    ));
    float3 v = make_float3(
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    );
    
    GPUCamera camera;
    camera.origin = lookfrom;
    camera.horizontal = float3_ops::mul(u, viewport_width);
    camera.vertical = float3_ops::mul(v, viewport_height);
    camera.lower_left = float3_ops::sub(
        float3_ops::sub(
            float3_ops::sub(lookfrom, float3_ops::mul(camera.horizontal, 0.5f)),
            float3_ops::mul(camera.vertical, 0.5f)
        ),
        float3_ops::mul(w, focal_length)
    );
    
    return camera;
}

int main(int argc, char* argv[]) {
    // Image settings
    const int width = 800;
    const int height = 600;
    const int max_bounces = 3;
    
    // CUDA device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
    std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock << " bytes\n";
    
    // Create scene data
    std::vector<GPUSphere> h_spheres;
    std::vector<GPULight> h_lights;
    
    // ===== TODO: STUDENT - CREATE GPU SCENE =====
    // Add spheres (aim for 50-100 for GPU testing)

    // Ground plane (large sphere below)
    h_spheres.push_back({
        make_float3(0, -1000, -20), 998.0f,
        {make_float3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f, 10.0f}
    });

    // Center spheres - different materials
    h_spheres.push_back({
        make_float3(0, 0, -20), 2.0f,
        {make_float3(1.0f, 0.3f, 0.3f), 0.0f, 0.8f, 10.0f}  // Diffuse red
    });

    h_spheres.push_back({
        make_float3(4, 0, -20), 1.5f,
        {make_float3(0.8f, 0.8f, 0.8f), 0.9f, 0.1f, 200.0f}  // Metallic
    });

    h_spheres.push_back({
        make_float3(-4, 0, -20), 1.5f,
        {make_float3(0.3f, 0.3f, 1.0f), 0.3f, 0.5f, 50.0f}  // Semi-metallic blue
    });

    // Create a grid of smaller spheres
    for (int i = -3; i <= 3; i++) {
        for (int j = -2; j <= 2; j++) {
            float x = i * 3.0f;
            float y = j * 2.5f + 1.0f;
            float z = -25.0f - (i * i + j * j) * 0.5f;

            // Vary colors based on position
            float r = (i + 3) / 6.0f;
            float g = (j + 2) / 4.0f;
            float b = 0.5f + 0.5f * ((i + j) % 2);

            // Vary metallicness
            float metallic = ((i + j) % 3) * 0.4f;

            h_spheres.push_back({
                make_float3(x, y, z), 0.8f,
                {make_float3(r, g, b), metallic, 0.5f, 30.0f}
            });
        }
    }

    // Add some random scattered spheres
    for (int i = 0; i < 20; i++) {
        float x = (i % 5 - 2) * 4.0f + (i % 3) * 0.5f;
        float y = (i % 4) * 0.8f;
        float z = -15.0f - (i % 7) * 2.0f;

        float r = (i % 5) / 4.0f;
        float g = ((i * 3) % 7) / 6.0f;
        float b = ((i * 7) % 11) / 10.0f;

        h_spheres.push_back({
            make_float3(x, y, z), 0.5f,
            {make_float3(r, g, b), (i % 3) * 0.3f, 0.6f, 20.0f}
        });
    }
    
    // Lights
    h_lights.push_back({
        make_float3(10, 10, -10),
        make_float3(1, 1, 1),
        0.7f
    });
    
    std::cout << "Scene: " << h_spheres.size() << " spheres, " 
              << h_lights.size() << " lights\n";
    
    // Allocate device memory
    GPUSphere* d_spheres;
    GPULight* d_lights;
    float3* d_framebuffer;
    
    CUDA_CHECK(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(GPUSphere)));
    CUDA_CHECK(cudaMalloc(&d_lights, h_lights.size() * sizeof(GPULight)));
    CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(float3)));
    
    // Copy scene to device
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(), 
                          h_spheres.size() * sizeof(GPUSphere),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lights, h_lights.data(),
                          h_lights.size() * sizeof(GPULight),
                          cudaMemcpyHostToDevice));
    
    // Setup camera
    GPUCamera camera = setup_camera(width, height);
    
    // Configure kernel launch
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);
    
    std::cout << "Launching kernel with " << blocks.x << "x" << blocks.y 
              << " blocks of " << threads.x << "x" << threads.y << " threads\n";
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Render
    std::cout << "Rendering..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    
    // ===== TODO: STUDENT - CHOOSE KERNEL VERSION =====
    // Start with basic kernel, then implement and test optimized version
    
    render_kernel<<<blocks, threads>>>(
        d_framebuffer, d_spheres, h_spheres.size(),
        d_lights, h_lights.size(), camera, width, height, max_bounces
    );
    
    // For optimized version with shared memory:
    // size_t shared_size = h_spheres.size() * sizeof(GPUSphere);
    // render_kernel_optimized<<<blocks, threads, shared_size>>>(...);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU rendering time: " << milliseconds / 1000.0f << " seconds\n";
    
    // Copy result back to host
    std::vector<float3> h_framebuffer(width * height);
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer,
                          width * height * sizeof(float3),
                          cudaMemcpyDeviceToHost));
    
    // Write output
    write_ppm("output_gpu.ppm", h_framebuffer, width, height);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_lights));
    CUDA_CHECK(cudaFree(d_framebuffer));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "Done! Output written to output_gpu.ppm\n";
    
    return 0;
}
