// main_hybrid.cpp - Week 3: Hybrid CPU-GPU Ray Tracer
// CS420 Ray Tracer Project

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =========================================================
// Camera Class (if you don't have camera.h)
// =========================================================
class Camera {
public:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    
    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect) {
        double theta = vfov * M_PI / 180.0;
        double h = tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect * viewport_height;
        
        Vec3 w = (lookfrom - lookat).normalized();
        Vec3 u = cross(vup, w).normalized();
        Vec3 v = cross(w, u);
        
        origin = lookfrom;
        horizontal = u * viewport_width;
        vertical = v * viewport_height;
        lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w;
    }
    
    Ray get_ray(double s, double t) const {
        Vec3 direction = lower_left_corner + horizontal * s + vertical * t - origin;
        return Ray(origin, direction.normalized());
    }
};

// =========================================================
// Image Output
// =========================================================
void write_ppm(const std::string& filename, const std::vector<Vec3>& framebuffer,
               int width, int height) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "P3\n" << width << " " << height << "\n255\n";
    
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            const Vec3& color = framebuffer[j * width + i];
            int r = static_cast<int>(255.99 * std::min(1.0, std::max(0.0, color.x)));
            int g = static_cast<int>(255.99 * std::min(1.0, std::max(0.0, color.y)));
            int b = static_cast<int>(255.99 * std::min(1.0, std::max(0.0, color.z)));
            file << r << " " << g << " " << b << "\n";
        }
    }
    
    file.close();
    std::cout << "Image written to " << filename << std::endl;
}

// =========================================================
// Scene Creation
// =========================================================
Scene create_test_scene() {
    Scene scene;
    
    // Red diffuse sphere (center)
    scene.spheres.push_back(Sphere(
        Vec3(0, 0, -20), 2.0,
        {Vec3(1, 0, 0), 0.0, 32.0}  // color, reflectivity, shininess
    ));
    
    // Silver metallic sphere (right)
    scene.spheres.push_back(Sphere(
        Vec3(3, 0, -20), 2.0,
        {Vec3(0.8, 0.8, 0.8), 0.8, 100.0}
    ));
    
    // Blue diffuse sphere (left)
    scene.spheres.push_back(Sphere(
        Vec3(-3, 0, -20), 2.0,
        {Vec3(0, 0, 1), 0.0, 32.0}
    ));
    
    // Ground sphere
    scene.spheres.push_back(Sphere(
        Vec3(0, -102, -20), 100.0,
        {Vec3(0.5, 0.5, 0.5), 0.0, 10.0}
    ));
    
    // Lights
    Light light1;
    light1.position = Vec3(10, 10, -10);
    light1.color = Vec3(1, 1, 1);
    light1.intensity = 0.7;
    scene.lights.push_back(light1);
    
    Light light2;
    light2.position = Vec3(-10, 10, -10);
    light2.color = Vec3(1, 1, 0.8);
    light2.intensity = 0.5;
    scene.lights.push_back(light2);
    
    scene.ambient_light = Vec3(0.1, 0.1, 0.1);
    
    return scene;
}

// =========================================================
// Tile Structure
// =========================================================
struct Tile {
    int x_start, y_start;
    int x_end, y_end;
    int complexity_estimate;
    bool processed;
    
    Tile(int xs, int ys, int xe, int ye)
        : x_start(xs), y_start(ys), x_end(xe), y_end(ye),
          complexity_estimate(0), processed(false) {}
    
    int width() const { return x_end - x_start; }
    int height() const { return y_end - y_start; }
    int pixel_count() const { return width() * height(); }
};

// =========================================================
// GPU Structures (must match main_gpu.cu)
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
};

// =========================================================
// GPU Kernel Declaration (from main_gpu.cu)
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
);

// =========================================================
// CPU Ray Tracing
// =========================================================
Vec3 trace_ray_cpu(const Ray& ray, const Scene& scene, int depth) {
    if (depth <= 0) return Vec3(0, 0, 0);
    
    double t;
    int sphere_idx;
    
    if (!scene.find_intersection(ray, t, sphere_idx)) {
        // Sky gradient
        double blend = 0.5 * (ray.direction.y + 1.0);
        return Vec3(1, 1, 1) * (1.0 - blend) + Vec3(0.5, 0.7, 1.0) * blend;
    }
    
    // Hit point and normal
    Vec3 hit_point = ray.at(t);
    const Sphere& sphere = scene.spheres[sphere_idx];
    Vec3 normal = sphere.normal_at(hit_point);
    Vec3 view_dir = (ray.origin - hit_point).normalized();
    
    // Shading
    const Material& mat = sphere.material;
    Vec3 color = scene.shade(hit_point, normal, mat, view_dir);
    
    // Reflections
    if (mat.reflectivity > 0.001 && depth > 1) {
        Vec3 incident = ray.direction.normalized();
        Vec3 reflect_dir = reflect(incident, normal);
        
        Vec3 offset_origin = hit_point + normal * 0.001;
        Ray reflect_ray(offset_origin, reflect_dir.normalized());
        
        Vec3 reflect_color = trace_ray_cpu(reflect_ray, scene, depth - 1);
        color = color * (1.0 - mat.reflectivity) + reflect_color * mat.reflectivity;
    }
    
    return color;
}

void process_tile_cpu(const Tile& tile, const Scene& scene, const Camera& camera,
                      std::vector<Vec3>& framebuffer, int width, int height, int max_depth) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int y = tile.y_start; y < tile.y_end; y++) {
        for (int x = tile.x_start; x < tile.x_end; x++) {
            double u = double(x) / double(width);
            double v = double(y) / double(height);
            Ray ray = camera.get_ray(u, v);
            framebuffer[y * width + x] = trace_ray_cpu(ray, scene, max_depth);
        }
    }
}

// =========================================================
// Complexity Estimation
// =========================================================
int estimate_tile_complexity(const Tile& tile, const Scene& scene, 
                             const Camera& camera, int width, int height) {
    int complexity = 0;
    const int STRIDE = 8;
    
    for (int y = tile.y_start; y < tile.y_end; y += STRIDE) {
        for (int x = tile.x_start; x < tile.x_end; x += STRIDE) {
            double u = double(x) / double(width);
            double v = double(y) / double(height);
            Ray ray = camera.get_ray(u, v);
            
            double t;
            int sphere_idx;
            if (scene.find_intersection(ray, t, sphere_idx)) {
                complexity += 10;
                if (scene.spheres[sphere_idx].material.reflectivity > 0.1) {
                    complexity += 20;  // Reflections are expensive
                }
            } else {
                complexity += 1;
            }
        }
    }
    
    return complexity;
}

// =========================================================
// GPU Resources Manager
// =========================================================
class GPUResources {
public:
    float3* d_framebuffer;
    GPUSphere* d_spheres;
    GPULight* d_lights;
    int num_spheres;
    int num_lights;
    
    GPUResources(int width, int height, const Scene& scene) {
        num_spheres = scene.spheres.size();
        num_lights = scene.lights.size();
        
        cudaMalloc(&d_framebuffer, width * height * sizeof(float3));
        cudaMalloc(&d_spheres, num_spheres * sizeof(GPUSphere));
        cudaMalloc(&d_lights, num_lights * sizeof(GPULight));
        
        // Convert and upload spheres
        std::vector<GPUSphere> gpu_spheres(num_spheres);
        for (int i = 0; i < num_spheres; i++) {
            const Sphere& s = scene.spheres[i];
            gpu_spheres[i].center = make_float3(s.center.x, s.center.y, s.center.z);
            gpu_spheres[i].radius = s.radius;
            gpu_spheres[i].material.albedo = make_float3(
                s.material.color.x, s.material.color.y, s.material.color.z);
            gpu_spheres[i].material.metallic = s.material.reflectivity;
            gpu_spheres[i].material.roughness = 1.0f - s.material.reflectivity;
            gpu_spheres[i].material.shininess = s.material.shininess;
        }
        cudaMemcpy(d_spheres, gpu_spheres.data(), 
                   num_spheres * sizeof(GPUSphere), cudaMemcpyHostToDevice);
        
        // Convert and upload lights
        std::vector<GPULight> gpu_lights(num_lights);
        for (int i = 0; i < num_lights; i++) {
            const Light& l = scene.lights[i];
            gpu_lights[i].position = make_float3(l.position.x, l.position.y, l.position.z);
            gpu_lights[i].color = make_float3(l.color.x, l.color.y, l.color.z);
            gpu_lights[i].intensity = l.intensity;
        }
        cudaMemcpy(d_lights, gpu_lights.data(),
                   num_lights * sizeof(GPULight), cudaMemcpyHostToDevice);
    }
    
    ~GPUResources() {
        cudaFree(d_framebuffer);
        cudaFree(d_spheres);
        cudaFree(d_lights);
    }
    
    void download_framebuffer(std::vector<float3>& buffer, int width, int height) {
        cudaMemcpy(buffer.data(), d_framebuffer,
                   width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    }
};

GPUCamera convert_camera(const Camera& camera) {
    GPUCamera gpu_cam;
    gpu_cam.origin = make_float3(camera.origin.x, camera.origin.y, camera.origin.z);
    gpu_cam.lower_left = make_float3(
        camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z);
    gpu_cam.horizontal = make_float3(
        camera.horizontal.x, camera.horizontal.y, camera.horizontal.z);
    gpu_cam.vertical = make_float3(
        camera.vertical.x, camera.vertical.y, camera.vertical.z);
    return gpu_cam;
}

// =========================================================
// Hybrid Rendering
// =========================================================
void render_hybrid(const Scene& scene, const Camera& camera,
                   std::vector<Vec3>& framebuffer,
                   int width, int height, int max_depth,
                   int tile_size = 64) {
    
    std::cout << "Hybrid Rendering..." << std::endl;
    
    // Create tiles
    std::vector<Tile> tiles;
    for (int y = 0; y < height; y += tile_size) {
        for (int x = 0; x < width; x += tile_size) {
            int xe = std::min(x + tile_size, width);
            int ye = std::min(y + tile_size, height);
            tiles.emplace_back(x, y, xe, ye);
        }
    }
    
    std::cout << "Created " << tiles.size() << " tiles of size " 
              << tile_size << "x" << tile_size << std::endl;
    
    // Estimate complexity
    for (auto& tile : tiles) {
        tile.complexity_estimate = estimate_tile_complexity(tile, scene, camera, width, height);
    }
    
    // Calculate threshold (average complexity)
    double avg_complexity = 0;
    for (const auto& tile : tiles) {
        avg_complexity += tile.complexity_estimate;
    }
    avg_complexity /= tiles.size();
    
    // Partition tiles: complex → CPU, simple → GPU
    std::vector<Tile*> cpu_tiles, gpu_tiles;
    for (auto& tile : tiles) {
        if (tile.complexity_estimate > avg_complexity * 1.3) {
            cpu_tiles.push_back(&tile);
        } else {
            gpu_tiles.push_back(&tile);
        }
    }
    
    std::cout << "Distribution: " << cpu_tiles.size() << " tiles to CPU, "
              << gpu_tiles.size() << " tiles to GPU" << std::endl;
    
    // Initialize GPU resources
    GPUResources gpu(width, height, scene);
    GPUCamera gpu_camera = convert_camera(camera);
    
    // Create CUDA streams
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Thread-safe indices for work stealing
    std::atomic<size_t> cpu_idx{0};
    std::atomic<size_t> gpu_idx{0};
    
    #pragma omp parallel sections
    {
        // CPU processing section
        #pragma omp section
        {
            size_t idx;
            while ((idx = cpu_idx.fetch_add(1)) < cpu_tiles.size()) {
                Tile* tile = cpu_tiles[idx];
                process_tile_cpu(*tile, scene, camera, framebuffer, width, height, max_depth);
                tile->processed = true;
            }
        }
        
        // GPU processing section
        #pragma omp section
        {
            size_t idx;
            int stream_idx = 0;
            while ((idx = gpu_idx.fetch_add(1)) < gpu_tiles.size()) {
                Tile* tile = gpu_tiles[idx];
                
                launch_tile_kernel(
                    gpu.d_framebuffer,
                    gpu.d_spheres, gpu.num_spheres,
                    gpu.d_lights, gpu.num_lights,
                    gpu_camera,
                    tile->x_start, tile->y_start,
                    tile->width(), tile->height(),
                    width, height,
                    max_depth,
                    streams[stream_idx]
                );
                
                tile->processed = true;
                stream_idx = (stream_idx + 1) % NUM_STREAMS;
            }
            
            // Wait for all GPU work
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamSynchronize(streams[i]);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Hybrid rendering time: " << elapsed << " seconds" << std::endl;
    
    // Download GPU results and merge into framebuffer
    std::vector<float3> gpu_buffer(width * height);
    gpu.download_framebuffer(gpu_buffer, width, height);
    
    for (const Tile* tile : gpu_tiles) {
        for (int y = tile->y_start; y < tile->y_end; y++) {
            for (int x = tile->x_start; x < tile->x_end; x++) {
                int idx = y * width + x;
                framebuffer[idx] = Vec3(gpu_buffer[idx].x, gpu_buffer[idx].y, gpu_buffer[idx].z);
            }
        }
    }
    
    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Verify all tiles processed
    int unprocessed = 0;
    for (const auto& tile : tiles) {
        if (!tile.processed) unprocessed++;
    }
    if (unprocessed > 0) {
        std::cerr << "Warning: " << unprocessed << " tiles not processed!" << std::endl;
    }
}

// =========================================================
// Scene File Loading
// =========================================================
void load_scene_file(const std::string& filename, Scene& scene) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open scene file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "sphere") {
            double x, y, z, radius, r, g, b, metallic, roughness, shininess;
            if (iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess) {
                Material mat;
                mat.color = Vec3(r, g, b);
                mat.reflectivity = metallic;
                mat.shininess = shininess;
                scene.spheres.push_back(Sphere(Vec3(x, y, z), radius, mat));
            }
        } else if (type == "light") {
            double x, y, z, r, g, b, intensity;
            if (iss >> x >> y >> z >> r >> g >> b >> intensity) {
                Light light;
                light.position = Vec3(x, y, z);
                light.color = Vec3(r, g, b);
                light.intensity = intensity;
                scene.lights.push_back(light);
            }
        }
    }
}

// =========================================================
// Main
// =========================================================
int main(int argc, char* argv[]) {
    // Image settings
    const int width = 800;
    const int height = 600;
    const int max_depth = 3;
    int tile_size = 64;
    std::string scene_file = "";
    std::string output_file = "output_hybrid.ppm";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-t" || arg == "--tile-size") && i + 1 < argc) {
            tile_size = std::atoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [scene_file] [options]\n"
                      << "Options:\n"
                      << "  -t, --tile-size SIZE   Set tile size (default: 64)\n"
                      << "  -o, --output FILE      Output filename\n"
                      << "  -h, --help             Show this help\n";
            return 0;
        } else if (arg[0] != '-') {
            scene_file = arg;
        }
    }
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found. Cannot run hybrid version." << std::endl;
        return 1;
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << "Tile size: " << tile_size << "x" << tile_size << std::endl;
    
    // Create or load scene
    Scene scene;
    if (!scene_file.empty()) {
        std::cout << "Loading scene from: " << scene_file << std::endl;
        load_scene_file(scene_file, scene);
    }
    
    if (scene.spheres.empty()) {
        std::cout << "Using default test scene" << std::endl;
        scene = create_test_scene();
    }
    
    std::cout << "Scene: " << scene.spheres.size() << " spheres, "
              << scene.lights.size() << " lights" << std::endl;
    
    // Setup camera
    double aspect_ratio = double(width) / double(height);
    Vec3 lookfrom(0, 2, 5);
    Vec3 lookat(0, 0, -20);
    Vec3 vup(0, 1, 0);
    double vfov = 60.0;
    
    Camera camera(lookfrom, lookat, vup, vfov, aspect_ratio);
    
    // Allocate framebuffer
    std::vector<Vec3> framebuffer(width * height);
    
    // Render
    render_hybrid(scene, camera, framebuffer, width, height, max_depth, tile_size);
    
    // Write output
    write_ppm(output_file, framebuffer, width, height);
    
    return 0;
}