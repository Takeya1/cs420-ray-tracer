// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
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
    
    //===== TODO: STUDENT - IMPLEMENT GPU SPHERE INTERSECTION =====
    __device__ bool intersect(const GPURay& ray, float t_min, float t_max, float& t) const {
        // TODO: Implement ray-sphere intersection on GPU
        // Same algorithm as CPU version but using float3 operations
        float3 oc = float3_ops::sub(ray.origin, center);
        float a = float3_ops::dot(ray.direction, ray.direction);
        float b = float3_ops::dot(oc, ray.direction);
        float c = float3_ops::dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        //check if discriminant is positive
        if (discriminant < 0) return false;
        else {
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
                             int max_bounces,
                             int samples_per_pixel) {

    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Accumulate color from multiple samples for antialiasing
    float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_per_pixel; s++) {
        // Compute sub-pixel offset for antialiasing (2x2 grid pattern)
        float dx = (s % 2) * 0.5f;
        float dy = (s / 2) * 0.5f;

        // TODO: STUDENT CODE HERE
        // Steps:
        // 1. Generate ray for this pixel
        float u = float(x + dx) / float(width);
        float v = float(y + dy) / float(height);
        GPURay ray = camera.get_ray(u, v);
        // 2. Initialize color accumulator and attenuation
        float3 sample_color = make_float3(0.0f, 0.0f, 0.0f);
        float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
        // 3. Iterative ray bouncing (instead of recursion):
        for (int bounce = 0; bounce < max_bounces; bounce++) {
    //        a. Find intersection
            float t_closest = 1e20f;
            GPUSphere* hit_sphere = nullptr;
            for (int i = 0; i < num_spheres; i++) {
                float t;
                if (spheres[i].intersect(ray, 0.001f, t_closest, t)) {
                    t_closest = t;
                    hit_sphere = &spheres[i];
                }
            }
            //        b. If no hit, add background color and break
            if (hit_sphere == nullptr) {
                float3 unit_direction = float3_ops::normalize(ray.direction);
                float t = 0.5f * (unit_direction.y + 1.0f);
                float3 background = float3_ops::lerp(make_float3(1.0f, 1.0f, 1.0f),
                                                    make_float3(0.5f, 0.7f, 1.0f), t);
                sample_color = float3_ops::add(sample_color,
                                              float3_ops::mul_componentwise(background, attenuation));
                break;
            }
    //        c. Calculate shading (ambient + diffuse + specular)
            float3 hit_point = ray.at(t_closest);
            float3 normal = hit_sphere->normal_at(hit_point);
            float3 view_dir = float3_ops::mul(ray.direction, -1.0f);
            
            // Simple ambient
            float3 ambient = float3_ops::mul(hit_sphere->material.albedo, 0.1f);
            float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
            float3 specular = make_float3(0.0f, 0.0f, 0.0f);
            
            for (int l = 0; l < num_lights; l++) {
                float3 light_dir = float3_ops::sub(lights[l].position, hit_point);
                light_dir = float3_ops::normalize(light_dir);
                
                // Diffuse
                float diff = fmaxf(float3_ops::dot(normal, light_dir), 0.0f);
                diffuse = float3_ops::add(diffuse, 
                                          float3_ops::mul(
                                              float3_ops::mul_componentwise(hit_sphere->material.albedo, lights[l].color),
                                              diff * lights[l].intensity));
                
                // Specular
                float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
                float spec = powf(fmaxf(float3_ops::dot(view_dir, reflect_dir), 0.0f), 
                                  hit_sphere->material.shininess);
                specular = float3_ops::add(specular, 
                                           float3_ops::mul(
                                               lights[l].color,
                                               spec * lights[l].intensity));
            }
            
            float3 local_color = float3_ops::add(ambient,
                                                 float3_ops::add(diffuse, specular));
            sample_color = float3_ops::add(sample_color,
                                          float3_ops::mul_componentwise(local_color, attenuation));

            // Prepare for next bounce
            ray.origin = hit_point;
            ray.direction = float3_ops::reflect(ray.direction, normal);
            attenuation = float3_ops::mul_componentwise(attenuation, hit_sphere->material.albedo);
        }
        //        d. If reflective, setup ray for next bounce
        //        e. Accumulate color with attenuation
        //    }

        // Accumulate this sample
        pixel_color = float3_ops::add(pixel_color, sample_color);
    }

    // Average all samples
    pixel_color = float3_ops::mul(pixel_color, 1.0f / float(samples_per_pixel));

    // 4. Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = pixel_color;
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
    //    extern __shared__ GPUSphere shared_spheres[]
    extern __shared__ GPUSphere shared_spheres[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    // 2. Cooperatively load spheres into shared memory
    for (int i = tid; i < num_spheres; i += total_threads) {
        shared_spheres[i] = global_spheres[i];
    }  
    // 3. __syncthreads()
    __syncthreads();
    // 4. Use shared_spheres instead of global_spheres for intersection tests
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    //generate ray (no antialiasing for simplicity)
    float u = float(x) / float(width);
    float v = float(y) / float(height);
    GPURay ray = camera.get_ray(u, v);
    
    float3 sample_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        float t_closest = 1e20f;
        int hit_idx = - 1; // use index instead of pointer
        //intersection test with shared memory spheres
        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (shared_spheres[i].intersect(ray, 0.001f, t_closest, t)) {
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
        // use shared_spheres[hit_idx] for shading
        float3 hit_point = ray.at(t_closest);
        float3 normal = shared_spheres[hit_idx].normal_at(hit_point);
        float3 view_dir = float3_ops::mul(ray.direction, -1.0f);
        // Simple ambient
        float3 ambient = float3_ops::mul(shared_spheres[hit_idx].material.albedo, 0.1f);
        float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
        float3 specular = make_float3(0.0f, 0.0f, 0.0f);

        for (int l = 0; l < num_lights; l++) {
            float3 light_dir = float3_ops::sub(lights[l].position, hit_point);
            light_dir = float3_ops::normalize(light_dir);
            
            // Diffuse
            float diff = fmaxf(float3_ops::dot(normal, light_dir), 0.0f);
            diffuse = float3_ops::add(diffuse, 
                                      float3_ops::mul(
                                          float3_ops::mul_componentwise(shared_spheres[hit_idx].material.albedo, lights[l].color),
                                          diff * lights[l].intensity));
            
            // Specular
            float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
            float spec = powf(fmaxf(float3_ops::dot(view_dir, reflect_dir), 0.0f), 
                              shared_spheres[hit_idx].material.shininess);
            specular = float3_ops::add(specular, 
                                       float3_ops::mul(
                                           lights[l].color,
                                           spec * lights[l].intensity));
        }
        float3 local_color = float3_ops::add(ambient,
                                             float3_ops::add(diffuse, specular));
        sample_color = float3_ops::add(sample_color,
                                      float3_ops::mul_componentwise(local_color, attenuation));

        // Prepare for next bounce
        ray.origin = hit_point;
        ray.direction = float3_ops::reflect(ray.direction, normal);
        attenuation = float3_ops::mul_componentwise(attenuation, shared_spheres[hit_idx].material.albedo);
    }
    // Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = sample_color;
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
    float3 u = float3_ops::normalize(float3_ops::cross(vup, w));
    float3 v = float3_ops::cross(w, u);
    
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

void load_scene_file(const std::string& filename,
                     std::vector<GPUSphere>& spheres,
                     std::vector<GPULight>& lights) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open scene file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "sphere") {
            // sphere: x y z radius r g b metallic roughness shininess
            float x, y, z, radius, r, g, b, metallic, roughness, shininess;
            if (iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess) {
                GPUSphere sphere;
                sphere.center = make_float3(x, y, z);
                sphere.radius = radius;
                sphere.material.albedo = make_float3(r, g, b);
                sphere.material.metallic = metallic;
                sphere.material.roughness = roughness;
                sphere.material.shininess = shininess;
                spheres.push_back(sphere);
            }
        } else if (type == "light") {
            // light: x y z r g b intensity
            float x, y, z, r, g, b, intensity;
            if (iss >> x >> y >> z >> r >> g >> b >> intensity) {
                GPULight light;
                light.position = make_float3(x, y, z);
                light.color = make_float3(r, g, b);
                light.intensity = intensity;
                lights.push_back(light);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Image settings
    const int width = 800;
    const int height = 600;
    const int max_bounces = 3;
    const int samples_per_pixel = 4;  // Antialiasing samples (4 = 2x2 grid)
    
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
    
    // ===== LOAD SCENE FROM FILE =====
    // Load scene from file (use scenes/simple.txt, scenes/medium.txt, or scenes/complex.txt)
    std::string scene_file = "scenes/simple.txt";
    if (argc > 1) {
        scene_file = argv[1];
    }
    
    std::cout << "Loading scene from: " << scene_file << std::endl;
    load_scene_file(scene_file, h_spheres, h_lights);
    
    // If no scene was loaded, use default spheres
    if (h_spheres.empty()) {
        std::cout << "No spheres loaded from file, using defaults\n";
        h_spheres.push_back({
            make_float3(0, 0, -20), 2.0f,
            {make_float3(1, 0, 0), 0.0f, 1.0f, 10.0f}
        });
        
        h_spheres.push_back({
            make_float3(3, 0, -20), 2.0f,
            {make_float3(0.8f, 0.8f, 0.8f), 0.8f, 0.2f, 100.0f}
        });
    }
    
    if (h_lights.empty()) {
        std::cout << "No lights loaded from file, using defaults\n";
        h_lights.push_back({
            make_float3(10, 10, -10),
            make_float3(1, 1, 1),
            0.7f
        });
    }
    
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
    std::cout << "Antialiasing: " << samples_per_pixel << " samples per pixel\n";

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Render
    std::cout << "Rendering..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    // ===== TODO: STUDENT - CHOOSE KERNEL VERSION =====
    // Start with basic kernel, then implement and test optimized version

    // render_kernel<<<blocks, threads>>>(
    //     d_framebuffer, d_spheres, (int)h_spheres.size(),
    //     d_lights, (int)h_lights.size(), camera, width, height, max_bounces,
    //     samples_per_pixel
    // );
    
    //For optimized version with shared memory:
    // For optimized version with shared memory:
    size_t shared_size = h_spheres.size() * sizeof(GPUSphere);
    render_kernel_optimized<<<blocks, threads, shared_size>>>(
        d_framebuffer, 
        d_spheres, (int)h_spheres.size(),
        d_lights, (int)h_lights.size(), 
        camera, 
        width, height, 
        max_bounces
    );
    
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
