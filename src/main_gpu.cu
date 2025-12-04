// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GPU-compatible Vec3 structure
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Camera configuration structure
struct CameraConfig {
    Vec3 position;
    Vec3 look_at;
    float fov;

    CameraConfig() : position(), look_at(0, 0, -1), fov(60.0f) {}
    CameraConfig(Vec3 pos, Vec3 target, float field_of_view)
        : position(pos), look_at(target), fov(field_of_view) {}
};

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
    __device__ static float3 make(float x, float y, float z) {
        return make_float3(x, y, z);
    }
    
    __device__ static float3 add(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    
    __device__ static float3 sub(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    
    __device__ static float3 mul(const float3& a, float t) {
        return make_float3(a.x * t, a.y * t, a.z * t);
    }
    
    __device__ static float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    
    __device__ static float length(const float3& v) {
        return sqrtf(dot(v, v));
    }
    
    __device__ static float3 normalize(const float3& v) {
        float len = length(v);
        return make_float3(v.x/len, v.y/len, v.z/len);
    }
    
    __device__ static float3 reflect(const float3& v, const float3& n) {
        return sub(v, mul(n, 2.0f * dot(v, n)));
    }
    
    __device__ static float3 lerp(const float3& a, const float3& b, float t) {
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
        // TODO: Implement ray-sphere intersection on GPU
        // Same algorithm as CPU version but using float3 operations
        
        // PLACEHOLDER
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
                             int max_bounces) {
    
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // TODO: STUDENT CODE HERE
    // Steps:
    // 1. Generate ray for this pixel
    // 2. Initialize color accumulator and attenuation
    // 3. Iterative ray bouncing (instead of recursion):
    //    for (int bounce = 0; bounce < max_bounces; bounce++) {
    //        a. Find intersection
    //        b. If no hit, add background color and break
    //        c. Calculate shading (ambient + diffuse + specular)
    //        d. If reflective, setup ray for next bounce
    //        e. Accumulate color with attenuation
    //    }
    // 4. Store final color in framebuffer
    
    // PLACEHOLDER - Just set to red for now
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = make_float3(1.0f, 0.0f, 0.0f);
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

// Load scene from text file (GPU-compatible version)
bool load_scene_gpu(const std::string& filename,
                    std::vector<GPUSphere>& spheres,
                    std::vector<GPULight>& lights,
                    float3& ambient_light,
                    CameraConfig& camera_config,
                    bool& has_camera) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open scene file: " << filename << std::endl;
        return false;
    }

    spheres.clear();
    lights.clear();
    ambient_light = make_float3(0.1f, 0.1f, 0.1f);
    has_camera = false;

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) {
            continue;
        }
        line = line.substr(start);

        // Skip lines that start with # after trimming
        if (line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sphere") {
            // sphere: x y z radius r g b metallic roughness shininess
            float x, y, z, radius;
            float r, g, b;
            float metallic, roughness, shininess;

            if (!(iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess)) {
                std::cerr << "Warning: Invalid sphere at line " << line_number << ", skipping\n";
                continue;
            }

            GPUSphere sphere;
            sphere.center = make_float3(x, y, z);
            sphere.radius = radius;
            sphere.material.albedo = make_float3(r, g, b);
            sphere.material.metallic = metallic;
            sphere.material.roughness = roughness;
            sphere.material.shininess = shininess;

            spheres.push_back(sphere);
        }
        else if (type == "light") {
            // light: x y z r g b intensity
            float x, y, z;
            float r, g, b;
            float intensity;

            if (!(iss >> x >> y >> z >> r >> g >> b >> intensity)) {
                std::cerr << "Warning: Invalid light at line " << line_number << ", skipping\n";
                continue;
            }

            GPULight light;
            light.position = make_float3(x, y, z);
            light.color = make_float3(r, g, b);
            light.intensity = intensity;

            lights.push_back(light);
        }
        else if (type == "ambient") {
            // ambient: r g b
            float r, g, b;

            if (!(iss >> r >> g >> b)) {
                std::cerr << "Warning: Invalid ambient at line " << line_number << ", skipping\n";
                continue;
            }

            ambient_light = make_float3(r, g, b);
        }
        else if (type == "camera") {
            // camera: pos_x pos_y pos_z look_x look_y look_z fov
            float px, py, pz;
            float lx, ly, lz;
            float fov;

            if (!(iss >> px >> py >> pz >> lx >> ly >> lz >> fov)) {
                std::cerr << "Warning: Invalid camera at line " << line_number << ", skipping\n";
                continue;
            }

            camera_config = CameraConfig(Vec3(px, py, pz), Vec3(lx, ly, lz), fov);
            has_camera = true;
        }
        else {
            std::cerr << "Warning: Unknown type '" << type << "' at line " << line_number << ", skipping\n";
        }
    }

    file.close();

    std::cout << "Loaded scene: " << spheres.size() << " spheres, "
              << lights.size() << " lights\n";

    return true;
}

GPUCamera setup_camera(int width, int height, const CameraConfig& config) {
    // Camera parameters from config
    float3 lookfrom = make_float3(config.position.x, config.position.y, config.position.z);
    float3 lookat = make_float3(config.look_at.x, config.look_at.y, config.look_at.z);
    float3 vup = make_float3(0, 1, 0);
    float vfov = config.fov;
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

    // Scene file to load (default: simple.txt)
    std::string scene_file = "scenes/simple.txt";

    // Allow command-line scene selection
    if (argc > 1) {
        scene_file = argv[1];
    }

    std::cout << "=== GPU Ray Tracer ===\n";
    std::cout << "Scene file: " << scene_file << "\n";
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Max bounces: " << max_bounces << "\n\n";

    // CUDA device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
    std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock << " bytes\n\n";

    // Load scene from file
    std::vector<GPUSphere> h_spheres;
    std::vector<GPULight> h_lights;
    float3 ambient_light;
    CameraConfig camera_config;
    bool has_camera = false;

    if (!load_scene_gpu(scene_file, h_spheres, h_lights, ambient_light, camera_config, has_camera)) {
        std::cerr << "Failed to load scene file: " << scene_file << std::endl;
        return 1;
    }

    if (!has_camera) {
        std::cout << "No camera specified in scene file, using default\n";
        camera_config = CameraConfig(Vec3(0, 2, 5), Vec3(0, 0, -20), 60.0f);
    }

    std::cout << "Scene: " << h_spheres.size() << " spheres, "
              << h_lights.size() << " lights\n";
    std::cout << "Camera: pos=(" << camera_config.position.x << ", "
              << camera_config.position.y << ", " << camera_config.position.z << ") "
              << "look_at=(" << camera_config.look_at.x << ", "
              << camera_config.look_at.y << ", " << camera_config.look_at.z << ") "
              << "fov=" << camera_config.fov << "\n\n";
    
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
    GPUCamera camera = setup_camera(width, height, camera_config);
    
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
