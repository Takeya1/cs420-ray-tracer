# CLAUDE.md - AI Assistant Guide for CS420 Ray Tracer

## Project Overview

**Project Type:** Educational Ray Tracing Engine
**Language:** C++11 with CUDA support
**Purpose:** CS420 Computer Graphics course assignment
**Platform:** UNIX/Linux (primary), with legacy Windows compatibility
**Parallel Frameworks:** Serial, OpenMP, CUDA, Hybrid CPU+GPU

This is a **student assignment template** with incomplete implementations. Students must implement core ray tracing algorithms including:
- Ray-sphere intersection
- Phong shading model
- Shadow calculations
- OpenMP parallelization
- CUDA GPU acceleration

## Repository Structure

```
cs420-ray-tracer/
├── include/                    # Header files
│   ├── vec3.h                 # 3D vector math (COMPLETE)
│   ├── ray.h                  # Ray class (COMPLETE)
│   ├── camera.h               # Camera from Ray Tracing in One Weekend (COMPLETE but complex)
│   ├── sphere.h               # Sphere geometry (TEMPLATE - needs intersection)
│   ├── material.h             # Phong shading (TEMPLATE - needs implementation)
│   ├── scene.h                # Scene management (TEMPLATE - needs shading logic)
│   └── math_constants.h       # M_PI for legacy Windows compatibility
│
├── src/                       # Source files
│   ├── main.cpp              # Week 1: Serial & OpenMP (TEMPLATE)
│   ├── main_gpu.cu           # Week 2: CUDA version (TEMPLATE)
│   └── main_hybrid.cpp       # Week 3: Hybrid CPU+GPU (TEMPLATE)
│
├── scenes/                    # Test scenes
│   ├── simple.txt            # 5 spheres, 2 lights
│   ├── medium.txt            # 50 spheres
│   └── complex.txt           # 200 spheres
│
├── scripts/                   # Automation
│   ├── test.sh               # Automated testing suite
│   └── benchmark.sh          # Performance comparison
│
├── makefile                   # Build system
├── README.md                  # Student instructions
└── UNIX_PACKAGE_FINAL.md     # UNIX/Linux deployment guide
```

## Core Architecture

### 1. Vector Mathematics (`vec3.h`)

**Status:** Complete and functional
**Location:** `include/vec3.h:1-32`

```cpp
class Vec3 {
    double x, y, z;
    // Supports: +, -, *, /, length(), normalized()
    // Helper functions: dot(), cross()
}
```

**Key Points:**
- Header-only implementation
- All operations inline for performance
- Used for positions, directions, colors

### 2. Ray Class (`ray.h`)

**Status:** Complete and functional
**Location:** `include/ray.h:1-17`

```cpp
class Ray {
    Vec3 origin;
    Vec3 direction;  // Always normalized
    Vec3 at(double t) const;  // Returns origin + direction * t
}
```

### 3. Sphere Geometry (`sphere.h`)

**Status:** TEMPLATE - Requires student implementation
**Location:** `include/sphere.h:1-40`

**TODO:** Students must implement `intersect()` method using quadratic equation:
- Solve: `||ray.origin + t*ray.direction - center||^2 = radius^2`
- Return smallest positive `t` value

**Material Conflict:** Note that `sphere.h` defines a simplified `Material` struct that conflicts with the more comprehensive `Material` class in `material.h`. This may need reconciliation.

### 4. Material and Shading (`material.h`)

**Status:** TEMPLATE - Requires Phong shading implementation
**Location:** `include/material.h:1-114`

**Material Types:**
- `DIFFUSE`: Lambertian diffuse surfaces
- `METAL`: Reflective metallic surfaces
- `DIELECTRIC`: Glass/transparent (extra credit)

**TODO:** Students must implement `shade()` method in `material.h:72-101`:
```cpp
Vec3 shade(const Vec3& hit_point, const Vec3& normal, const Vec3& view_dir,
           const Vec3& light_pos, const Vec3& light_color,
           const Vec3& ambient) const
```

**Phong Model Formula:**
```
I_total = I_ambient + I_diffuse + I_specular
I_ambient = k_a * ambient_light_color
I_diffuse = k_d * light_color * max(0, N·L)
I_specular = k_s * light_color * max(0, R·V)^n
```

### 5. Scene Management (`scene.h`)

**Status:** TEMPLATE - Requires implementation
**Location:** `include/scene.h:1-60`

**Components:**
- `std::vector<Sphere> spheres`
- `std::vector<Light> lights`
- `Vec3 ambient_light`

**TODO:** Students must implement:
1. `find_intersection()` - Loop through spheres, find closest hit
2. `in_shadow()` - Cast shadow ray to light source
3. `shade()` - Apply Phong shading with shadow tests

### 6. Camera System (`camera.h`)

**Status:** Complete but complex (from Ray Tracing in One Weekend)
**Location:** `include/camera.h:1-154`

**Warning:** This file uses advanced features:
- Depends on undefined types: `hittable`, `hit_record`, `point3`, `color`
- Uses depth of field and antialiasing
- May not be compatible with simple student assignment

**Recommendation:** Students use simplified camera from `main.cpp:12-35` instead.

### 7. Main Rendering Loop (`main.cpp`)

**Status:** TEMPLATE with structure provided
**Location:** `src/main.cpp:1-140`

**Key Functions:**
- `trace_ray()` - Recursive ray tracing (needs implementation at line 38-57)
- `write_ppm()` - Output PPM image format (complete)
- `main()` - Setup and render loop (structure provided)

**TODO Areas:**
1. **Line 38-57:** Implement ray tracing logic
2. **Line 85-92:** Add scene spheres and lights
3. **Line 124-137:** Add OpenMP parallelization

### 8. CUDA Version (`main_gpu.cu`)

**Status:** Week 2 template
**Location:** `src/main_gpu.cu`

Students implement GPU kernels for ray tracing with CUDA.

### 9. Hybrid Version (`main_hybrid.cpp`)

**Status:** Week 3 template
**Location:** `src/main_hybrid.cpp`

Students combine OpenMP CPU parallelization with CUDA GPU acceleration.

## Build System

### Makefile Structure

**Location:** `makefile:1-44`

**Build Targets:**
```bash
make serial       # Week 1: Serial CPU version
make openmp       # Week 1: OpenMP parallel version
make cuda         # Week 2: CUDA GPU version
make hybrid       # Week 3: Hybrid CPU+GPU version
make all          # Build all versions
```

**Testing:**
```bash
make test         # Quick test with serial version
make benchmark    # Compare serial vs OpenMP performance
```

**Cleanup:**
```bash
make clean        # Remove all build artifacts
```

**Compiler Configuration:**
- C++ Compiler: `g++` with `-std=c++11 -O3 -Wall`
- OpenMP: `-fopenmp`
- CUDA: `nvcc` with `-O3 -arch=sm_60`
- Include path: `-I$(INCDIR)` for `include/` directory

### Build Dependencies

**Required:**
- g++ or clang++ (C++11 support)
- make
- OpenMP library (libomp-dev on Ubuntu)

**Optional:**
- CUDA Toolkit (for GPU version)
- valgrind (for memory checking)
- perf (for profiling)
- ImageMagick (for PPM to PNG conversion)

## Development Workflows

### Week 1: Serial and OpenMP

**Goals:**
1. Implement ray-sphere intersection (`sphere.h:25-33`)
2. Implement Phong shading (`material.h:72-101`)
3. Implement scene intersection and shading (`scene.h`)
4. Complete trace_ray function (`main.cpp:38-57`)
5. Add OpenMP parallelization (`main.cpp:124-137`)

**Performance Target:** 2.5x speedup with OpenMP

**Testing:**
```bash
make serial
./ray_serial
# Should generate output_serial.ppm

make openmp
./ray_openmp
# Should generate output_openmp.ppm with 2.5x+ speedup
```

### Week 2: CUDA GPU

**Goals:**
1. Port ray tracing to CUDA kernels
2. Manage GPU memory allocation
3. Optimize thread block configuration

**Performance Target:** 10x speedup over serial

**Build:**
```bash
make cuda
./ray_cuda
```

### Week 3: Hybrid CPU+GPU

**Goals:**
1. Combine OpenMP and CUDA
2. Load balance between CPU and GPU
3. Minimize data transfer overhead

**Build:**
```bash
make hybrid
./ray_hybrid
```

## Scene File Format

**Location:** `scenes/simple.txt:1-22`

```
# Sphere format
sphere x y z radius r g b metallic roughness shininess

# Light format
light x y z r g b intensity

# Ambient light
ambient r g b

# Camera
camera pos_x pos_y pos_z lookat_x lookat_y lookat_z fov
```

**Example:**
```
sphere 0 0 -20 2 1.0 0.0 0.0 0.0 1.0 10
light 10 10 -10 1.0 1.0 1.0 0.7
ambient 0.1 0.1 0.1
camera 0 2 5 0 0 -20 60
```

## Key Implementation Conventions

### Coordinate System

- **Right-handed:** X-right, Y-up, Z-forward (towards camera)
- **Camera looks:** Down negative Z-axis
- **Scene depth:** Objects at negative Z values (e.g., -20)

### Color Representation

- **Storage:** `Vec3` with components in [0, 1]
- **Output:** PPM format with RGB values [0, 255]
- **Gamma:** No gamma correction in template (linear color space)

### Ray Tracing Algorithm

**Pseudocode:**
```
function trace_ray(ray, scene, depth):
    if depth <= 0:
        return black

    if ray hits scene:
        hit_point = ray.at(t)
        normal = sphere.normal_at(hit_point)
        color = scene.shade(hit_point, normal, material, view_dir)

        if material.is_reflective():
            reflect_ray = create_reflection(ray, normal, hit_point)
            reflect_color = trace_ray(reflect_ray, scene, depth-1)
            color = mix(color, reflect_color, material.reflectivity)

        return color
    else:
        return sky_gradient(ray.direction.y)
```

### Performance Considerations

**OpenMP Best Practices:**
```cpp
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
        // Render pixel (i, j)
    }
}
```

- Use `schedule(dynamic)` for load balancing
- Each pixel is independent (embarrassingly parallel)
- Watch out for race conditions when writing to framebuffer

**CUDA Optimization:**
- 1 thread per pixel is typical
- Block size: 16x16 or 32x32 threads
- Minimize host-device transfers
- Use constant memory for scene data if possible

## Testing and Validation

### Automated Testing

**Script:** `scripts/test.sh`

```bash
# Run all tests
./scripts/test.sh all

# Test specific version
./scripts/test.sh serial
./scripts/test.sh openmp
./scripts/test.sh cuda
```

**Tests Performed:**
- Build verification
- Runtime execution (30s timeout)
- Output file generation
- File size validation
- Performance measurement
- Image comparison between versions

### Benchmarking

**Script:** `scripts/benchmark.sh`

Compares performance across all implementations:
- Serial baseline
- OpenMP speedup (target: 2.5x)
- CUDA speedup (target: 10x)

### Expected Output

**Visual Verification:**
- Scene with colored spheres
- Proper lighting and shadows
- Reflections on metallic spheres
- Gradient sky background

**Invalid Output Indicators:**
- All black image → No intersections found
- All magenta → Placeholder color from line 56
- Missing shadows → Shadow calculation incomplete
- No reflections → Reflective ray tracing not implemented

## Common Pitfalls and Solutions

### 1. Material Struct Conflict

**Problem:** Two different `Material` definitions:
- `sphere.h:7-11` - Simple struct
- `material.h:20-112` - Complete class

**Solution:** Use `material.h` version. Consider removing simplified struct from `sphere.h`.

### 2. Camera Include Issues

**Problem:** `camera.h:14-15` includes undefined headers:
```cpp
#include "hittable.h"  // NOT PRESENT
#include "material.h"
```

**Solution:** Don't use advanced `camera.h`. Use simplified camera in `main.cpp:12-35`.

### 3. Math Constants on Windows

**Problem:** `M_PI` not defined on Windows without `_USE_MATH_DEFINES`

**Current State:** `math_constants.h` exists but deprecated for UNIX

**Solution:** Code uses `M_PI` directly (line 27 in `main.cpp`). Works on UNIX. For Windows compatibility, include `<cmath>` and use `M_PI` or define manually.

### 4. OpenMP Conditional Compilation

**Problem:** OpenMP code should only compile when `-fopenmp` flag is used

**Solution:** Wrap OpenMP code:
```cpp
#ifdef _OPENMP
    // OpenMP code here
#endif
```

### 5. Ray Direction Normalization

**Important:** Ray directions are **always normalized** in constructor (`ray.h:12`)

**Impact:** Don't re-normalize ray directions in calculations.

### 6. Scene File Loading

**Problem:** Template doesn't include scene file parser

**Status:** Students must either:
- Hard-code scenes in `main.cpp:85-92`
- Implement scene file parser (extra credit)

### 7. Reflection Ray Offset

**Problem:** Reflected rays starting exactly on surface cause self-intersection

**Solution:** Offset ray origin slightly along normal:
```cpp
Vec3 reflect_origin = hit_point + normal * 0.001;
```

## Code Quality Standards

### Naming Conventions

- **Classes:** PascalCase (`Vec3`, `Ray`, `Sphere`)
- **Functions:** snake_case (`trace_ray`, `find_intersection`)
- **Variables:** snake_case (`hit_point`, `light_dir`)
- **Constants:** UPPER_CASE (`M_PI`)

### Header Guards

All headers use `#ifndef` guards:
```cpp
#ifndef VEC3_H
#define VEC3_H
// ...
#endif
```

### Documentation Style

Student templates use `TODO:` comments:
```cpp
// TODO: STUDENT IMPLEMENTATION
// Implement ray-sphere intersection test
```

## AI Assistant Best Practices

### When Helping with Implementation

1. **Identify Assignment Context:** This is student coursework. Provide guidance, not complete solutions.

2. **Check Implementation Status:**
   - `vec3.h`, `ray.h` → Complete, don't modify
   - `sphere.h`, `material.h`, `scene.h`, `main.cpp` → Templates, students must complete

3. **Preserve TODO Comments:** Keep student instructions intact

4. **Test Before Committing:**
   ```bash
   make clean
   make serial
   ./ray_serial
   # Verify output_serial.ppm is generated and looks correct
   ```

5. **Verify Build System:**
   - Don't break existing Makefile targets
   - Test both `serial` and `openmp` builds
   - Ensure CUDA builds are optional (may not have GPU)

### When Debugging

1. **Check Common Issues First:**
   - Material struct conflict
   - Missing intersections (all black image)
   - Incorrect normal calculation (inverted lighting)
   - Shadow ray self-intersection
   - Missing `#ifdef _OPENMP` guards

2. **Use Placeholder Colors:**
   - Normals: `return (normal + Vec3(1,1,1)) * 0.5` (visualize normals)
   - Depth: `return Vec3(t/20, t/20, t/20)` (visualize distance)
   - Material ID: Different colors per sphere

3. **Progressive Implementation:**
   - Step 1: Just intersection (colored spheres, no lighting)
   - Step 2: Add ambient lighting
   - Step 3: Add diffuse (Lambert)
   - Step 4: Add specular (Phong)
   - Step 5: Add shadows
   - Step 6: Add reflections

### When Modifying Files

1. **Read Before Writing:** Always read files before editing
2. **Preserve Structure:** Keep TODO sections intact
3. **Match Style:** Follow existing code conventions
4. **Test Incrementally:** Build and test after each change
5. **Document Changes:** Add comments explaining modifications

### When Creating New Features

1. **Check Assignment Scope:** Don't add features beyond requirements
2. **Maintain Compatibility:** Don't break existing tests
3. **Update Documentation:** Add to this CLAUDE.md if significant

## Git Workflow

**Current Branch:** `claude/claude-md-mi7r9crsu2bz7vhm-01VCLCHNZ3a6tgGsBPRNEXez`

**When Committing:**
1. Build and test first
2. Use descriptive commit messages
3. Don't commit build artifacts (*.o, *.ppm, executables)
4. Commit related changes together

**Protected Files:**
- `.gitignore` - Ensure build artifacts excluded
- `makefile` - Only modify if adding new targets
- Complete headers (`vec3.h`, `ray.h`) - Don't modify

## Performance Metrics

### Expected Runtimes (640x480 resolution)

**Serial (baseline):**
- Simple scene (5 spheres): ~2-5 seconds
- Medium scene (50 spheres): ~20-40 seconds
- Complex scene (200 spheres): ~60-120 seconds

**OpenMP (4 threads):**
- Target: 2.5x speedup minimum
- Achievable: 3-3.5x speedup

**CUDA (GPU):**
- Target: 10x speedup minimum
- Achievable: 20-50x speedup (depending on hardware)

## Additional Resources

### Ray Tracing References

- **Ray Tracing in One Weekend:** Source for camera.h design
- **Phong Reflection Model:** Classic illumination algorithm
- **Scratchapixel:** Excellent ray tracing tutorials

### Parallel Programming

- **OpenMP:** Task-based parallelism for CPUs
- **CUDA:** GPU programming model from NVIDIA

### Output Format

- **PPM (Portable Pixmap):** Simple uncompressed image format
- **Conversion:** Use ImageMagick `convert` to PNG/JPEG

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| All black image | No intersections found | `sphere.h:25-33` intersection |
| All magenta | Using placeholder return | `main.cpp:56` |
| No lighting | Shading not implemented | `material.h:72-101` |
| Harsh edges | No antialiasing | Expected in basic version |
| Self-shadowing artifacts | Ray offset too small | Add `epsilon = 0.001` offset |
| Compile error: M_PI | Windows compatibility | Include `<cmath>` or define manually |
| Compile error: hittable.h | Using advanced camera.h | Use simple camera in main.cpp |
| OpenMP slower than serial | Overhead too high | Use larger scenes or reduce thread count |
| CUDA won't build | No CUDA toolkit | Install CUDA or skip GPU targets |

## File Modification Safety Matrix

| File | Safe to Modify | Notes |
|------|---------------|-------|
| `vec3.h` | ⚠️ Caution | Complete implementation, only fix bugs |
| `ray.h` | ⚠️ Caution | Complete implementation, only fix bugs |
| `sphere.h` | ✅ Yes | Student must implement `intersect()` |
| `material.h` | ✅ Yes | Student must implement `shade()` |
| `scene.h` | ✅ Yes | Student must implement all methods |
| `camera.h` | ❌ No | Complex, likely incompatible, use simple camera |
| `math_constants.h` | ⚠️ Caution | Deprecated for UNIX, keep for Windows compat |
| `main.cpp` | ✅ Yes | Primary student implementation file |
| `main_gpu.cu` | ✅ Yes | Week 2 CUDA implementation |
| `main_hybrid.cpp` | ✅ Yes | Week 3 hybrid implementation |
| `makefile` | ⚠️ Caution | Functional, only modify for new features |
| `scripts/*.sh` | ⚠️ Caution | Testing infrastructure, modify carefully |
| Scene files | ✅ Yes | Safe to create new test scenes |

## Version Information

- **C++ Standard:** C++11
- **CUDA Compute Capability:** sm_60 (Pascal architecture minimum)
- **OpenMP Version:** 4.0+ (implicit from g++ support)
- **Build System:** GNU Make

---

**Last Updated:** 2025-11-20
**For:** CS420 Computer Graphics Course
**Maintainer:** Course Instructor
**AI Assistant Version:** Optimized for Claude Code assistance
