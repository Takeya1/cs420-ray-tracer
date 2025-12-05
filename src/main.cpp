#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include "scene_loader.h"
#include "ray_constants.h"

class Camera {
public:
        Vec3 position;
        Vec3 forward, right, up;
        double fov;
        
        Camera(Vec3 pos, Vec3 look_at, double field_of_view) 
            : position(pos), fov(field_of_view) {
            forward = (look_at - position).normalized();
            right = cross(forward, Vec3(0, 1, 0)).normalized();
            up = cross(right, forward).normalized();
        }
        
        Ray get_ray(double u, double v) const {
            double aspect = 1.0;
            double scale = tan(fov * 0.5 * M_PI / 180.0);
            
            Vec3 direction = forward 
                           + right * ((u - 0.5) * scale * aspect)
                           + up * ((v - 0.5) * scale);
            
            return Ray(position, direction.normalized());
        }
};

// Trace a single ray through the scene
Vec3 trace_ray(const Ray& ray, const Scene& scene, int depth) {
    if (depth <= 0) return Vec3(0, 0, 0);
    
    double t;
    int sphere_idx;
    
    if (!scene.find_intersection(ray, t, sphere_idx)) {
        // Sky color gradient
        double t = 0.5 * (ray.direction.y + 1.0);
        return Vec3(1, 1, 1) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
    }
    
    // TODO: STUDENT IMPLEMENTATION
    // 1. Calculate hit point and normal
    // 2. Call scene.shade() for color
    // 3. If material is reflective, recursively trace reflection ray
    // YOUR CODE HERE
    Vec3 hit_point = ray.at(t);
    
    // Get the sphere and calculate normal
    const Sphere& sphere = scene.spheres[sphere_idx];
    Vec3 normal = sphere.normal_at(hit_point);
    
    // Calculate view direction (from hit point to camera)
    Vec3 view_dir = (ray.origin - hit_point).normalized();
    
    // Get material and calculate base color with shading
    const Material& mat = sphere.material;
    Vec3 color = scene.shade(hit_point, normal, mat, view_dir);
    
    // Handle reflections if material is reflective
    if (mat.reflectivity > 0.0 && depth > 1) {
        // Calculate reflection direction: R = I - 2*(I·N)*N
        // where I is the incident direction (ray.direction)
        Vec3 incident = ray.direction.normalized();
        Vec3 reflect_dir = incident - normal * (2.0 * dot(incident, normal));
        
        // Offset reflection ray origin to avoid self-intersection
        Vec3 offset_origin = hit_point + normal * EPSILON;
        Ray reflect_ray(offset_origin, reflect_dir.normalized());
        
        // Recursively trace reflection ray
        Vec3 reflect_color = trace_ray(reflect_ray, scene, depth - 1);
        
        // Blend base color with reflection
        color = color * (1.0 - mat.reflectivity) + reflect_color * mat.reflectivity;
    }
    
    return color;
}

// Write image to PPM file
void write_ppm(const std::string& filename, const std::vector<Vec3>& framebuffer, 
               int width, int height) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            Vec3 color = framebuffer[j * width + i];
            int r = int(255.99 * std::min(1.0, color.x));
            int g = int(255.99 * std::min(1.0, color.y));
            int b = int(255.99 * std::min(1.0, color.z));
            file << r << " " << g << " " << b << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    // Image settings
    const int width = 640;
    const int height = 480;
    const int max_depth = 3;
    const int samples_per_pixel = 4;  // Antialiasing samples (4 = 2x2 grid)

    // Scene file to load (default: simple.txt)
    std::string scene_file = "scenes/simple.txt";

    // Allow command-line scene selection
    if (argc > 1) {
        scene_file = argv[1];
    }
    
    // Load scene from file
    std::cout << "Loading scene from: " << scene_file << "\n";
    SceneData scene_data = load_scene(scene_file);
    Scene scene = scene_data.scene;
    
    // Print scene info for debugging
    print_scene_info(scene_data);
    
    // Setup camera from loaded scene data (or use default if not specified)
    Camera camera = scene_data.has_camera 
        ? Camera(scene_data.camera.position, scene_data.camera.look_at, scene_data.camera.fov)
        : Camera(Vec3(0, 0, 0), Vec3(0, 0, -1), 60);
    
    if (scene_data.has_camera) {
        std::cout << "Using camera from scene file\n";
    } else {
        std::cout << "Using default camera\n";
    }
    
    // Framebuffer
    std::vector<Vec3> framebuffer(width * height);
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // SERIAL VERSION
    std::cout << "Rendering (Serial) with " << samples_per_pixel << " samples per pixel...\n";
    for (int j = 0; j < height; j++) {
        if (j % 50 == 0) std::cout << "Row " << j << "/" << height << "\n";

        for (int i = 0; i < width; i++) {
            Vec3 pixel_color(0, 0, 0);

            // Take multiple samples per pixel for antialiasing
            for (int s = 0; s < samples_per_pixel; s++) {
                // Random offset within the pixel for stochastic sampling
                // For deterministic 2x2 grid pattern:
                double dx = (s % 2) * 0.5;
                double dy = (s / 2) * 0.5;

                double u = (double(i) + dx) / (width - 1);
                double v = (double(j) + dy) / (height - 1);

                Ray ray = camera.get_ray(u, v);
                pixel_color = pixel_color + trace_ray(ray, scene, max_depth);
            }

            // Average the samples
            framebuffer[j * width + i] = pixel_color * (1.0 / samples_per_pixel);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Serial time: " << diff.count() << " seconds\n";
    
    write_ppm("output_serial.ppm", framebuffer, width, height);
    
    // TODO: STUDENT - Add OpenMP version
    // OPENMP VERSION
    #ifdef _OPENMP
    std::cout << "\nRendering (OpenMP) with " << samples_per_pixel << " samples per pixel...\n";
    start = std::chrono::high_resolution_clock::now();

    // YOUR OPENMP CODE HERE
    // Hint: Use #pragma omp parallel for with appropriate scheduling
    #pragma omp parallel for schedule (dynamic, 8)
    for (int j = 0; j < height; j++){
        for (int i = 0; i < width; i++){
            Vec3 pixel_color(0, 0, 0);

            // Take multiple samples per pixel for antialiasing
            for (int s = 0; s < samples_per_pixel; s++) {
                double dx = (s % 2) * 0.5;
                double dy = (s / 2) * 0.5;

                double u = (double(i) + dx) / (width - 1);
                double v = (double(j) + dy) / (height - 1);

                Ray ray = camera.get_ray(u, v);
                pixel_color = pixel_color + trace_ray(ray, scene, max_depth);
            }

            // Average the samples
            framebuffer[j * width + i] = pixel_color * (1.0 / samples_per_pixel);
        }
    }


    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "OpenMP time: " << diff.count() << " seconds\n";

    write_ppm("output_openmp.ppm", framebuffer, width, height);
    #endif
    
    return 0;
}