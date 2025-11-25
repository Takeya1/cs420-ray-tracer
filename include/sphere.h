#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"

struct Material {
    Vec3 color;           // Base color
    double reflectivity;  // 0.0 = diffuse, 1.0 = mirror
    double shininess;     // Phong exponent
};

class Sphere {
public:
    Vec3 center;
    double radius;
    Material material;
    
    Sphere(Vec3 c, double r, Material m) : center(c), radius(r), material(m) {}
    
    // TODO: STUDENT IMPLEMENTATION
    // Implement ray-sphere intersection test
    // Return true if ray hits sphere, store distance in t
    // Hint: Solve quadratic equation from ray equation and sphere equation
    bool intersect(const Ray& ray, double& t) const {
        // YOUR CODE HERE
        // 1. Calculate discriminant
        Vec3 oc = ray.origin - center;
        double a = dot(ray.direction, ray.direction);
        double b = 2.0 * dot(oc, ray.direction);
        double c = dot(oc, oc) - radius * radius;

        double discriminant = b*b - 4*a*c;
        // 2. Check if discriminant >= 0
        if (discriminant < 0) return false;
        // 3. Calculate t values
        else {
            double sqrt_discriminant = sqrt(discriminant);
            double t1 = (-b - sqrt_discriminant) / (2*a);
            double t2 = (-b + sqrt_discriminant) / (2*a);
            // 4. Return smallest positive t
            if (t1 > 0) t = t1;
            else if (t2 > 0) t = t2;
            else return false;
        }
        return true;
    }
    
    // Calculate normal at point on sphere surface
    Vec3 normal_at(const Vec3& point) const {
        return (point - center).normalized();
    }
};

#endif