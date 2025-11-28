#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <cmath>
#include "sphere.h"
#include "vec3.h"
#include "ray.h"
#include "math_constants.h"

struct Light {
    Vec3 position;
    Vec3 color;
    double intensity;
};

class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Light> lights;
    Vec3 ambient_light;
    
    Scene() : ambient_light(0.1, 0.1, 0.1) {}
    
    // Find closest sphere intersection
    bool find_intersection(const Ray& ray, double& t, int& sphere_idx) const {
        t = INFINITY_DOUBLE;
        sphere_idx = -1;
        
        // TODO: STUDENT IMPLEMENTATION
        // Loop through all spheres and find the closest intersection

        // YOUR CODE HERE
        for (size_t i = 0; i < spheres.size(); i++) {
            double t_temp;
            if(spheres[i].intersect(ray, t_temp)) {
                if(t_temp < t) {
                    t = t_temp;
                    sphere_idx = static_cast<int>(i);
                }
            }
        }
        
        return sphere_idx >= 0;
    }
    
    // Check if point is in shadow from light
    bool in_shadow(const Vec3& point, const Light& light) const {
        // TODO: STUDENT IMPLEMENTATION
        // Cast ray from point to light and check for intersections
        // YOUR CODE HERE
        Vec3 to_light = light.position - point;
        double distance_to_light = to_light.length();
        
        // Offset the ray origin slightly to avoid self-intersection
        Vec3 shadow_dir = to_light.normalized();
        Vec3 offset_origin = point + shadow_dir * EPSILON;
        Ray shadow_ray(offset_origin, shadow_dir);
        
        // Check for intersections
        double t;
        int sphere_idx;
        if (find_intersection(shadow_ray, t, sphere_idx)) {
            // If intersection is closer than the light, point is in shadow
            if (t > 0 && t < distance_to_light) {
                return true;
            }
        }
        
        return false;  // Not in shadow
    }
    
    // Calculate color at intersection point using Phong shading
    Vec3 shade(const Vec3& point, const Vec3& normal, const Material& mat, 
               const Vec3& view_dir) const {
        Vec3 color = ambient_light * mat.color;
        
        // TODO: STUDENT IMPLEMENTATION
        // For each light:
        //   1. Check if in shadow
        //   2. Calculate diffuse component (Lambert)
        //   3. Calculate specular component (Phong)
        // YOUR CODE HERE
        
        // Loop through all lights
        for (const Light& light : lights) {
            // 1. Check if point is in shadow from this light
            if (in_shadow(point, light)) {
                continue;  // Skip this light if in shadow
            }
            
            // 2. Calculate light direction
            Vec3 to_light = (light.position - point).normalized();
            
            // 3. Calculate diffuse component (Lambert's law)
            double n_dot_l = std::max(0.0, dot(normal, to_light));
            Vec3 diffuse = mat.color * light.color * light.intensity * n_dot_l;
            
            // 4. Calculate specular component (Phong reflection)
            Vec3 reflect_dir = normal * (2.0 * n_dot_l) - to_light;
            double r_dot_v = std::max(0.0, dot(reflect_dir, view_dir));
            double specular_factor = pow(r_dot_v, mat.shininess);
            Vec3 specular = light.color * light.intensity * specular_factor * (1.0 - mat.reflectivity);
            
            // Add diffuse and specular contributions
            color = color + diffuse + specular;
        }
        
        return color;
    }
};

#endif