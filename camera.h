#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera {
    public:
        Vector origin;
        Vector lower_left_corner;
        Vector horizontal;
        Vector vertical;
    
    
        __device__ Camera() {
            lower_left_corner = Vector(-2.0, -1.0, -1.0);
            horizontal        = Vector( 4.0,  0.0,  0.0);
            vertical          = Vector( 0.0,  2.0,  0.0);
            origin            = Vector( 0.0,  0.0,  0.0);
        }
    
        __device__ Ray get_ray(float u, float v) {
            return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
        }
};

#endif // CAMERA_H
