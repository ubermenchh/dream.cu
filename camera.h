#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include <curand_kernel.h>

__device__ Vector random_in_unit_disk(curandState* local_rand_state) {
    Vector p;
    do {
        p = 2.f * Vector(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vector(1, 1, 0);
    } while (dot(p, p) >= 1.f);
    return p;
}

class Camera {
    public:
        Vector origin;
        Vector lower_left_corner;
        Vector horizontal;
        Vector vertical;
        Vector u, v, w;
        double lens_radius;
    
        __device__ Camera(Vector look_from, Vector look_at, Vector v_up, double v_fov, double aspect, double aperture, double focus_dist) {
            // v_fov is top to bottom in degrees 
            double theta = v_fov * (double)M_PI / 180.f;
            double half_height = tan(theta / 2.f);
            double half_width = aspect * half_height;
            
            origin = look_from;
            w = unit_vector(look_from - look_at);
            u = unit_vector(cross(v_up, w));
            v = cross(w, u);
            
            lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
            horizontal = 2 * half_width * focus_dist * u;
            vertical = 2 * half_height * focus_dist * v;
        }
    
        __device__ Ray get_ray(float s, float t, curandState* local_rand_state) {
            Vector rd = lens_radius * random_in_unit_disk(local_rand_state);
            Vector offset = u * rd.x() + v * rd.y();
            return Ray(origin, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }
};

#endif // CAMERA_H
