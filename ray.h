#ifndef RAY_H 
#define RAY_H
 
#include "vector.h" 

class Ray {
    public:
        Vector origin;
        Vector direction;

        __device__ Ray() {}
        __device__ Ray(const Vector& origin, const Vector& direction) { 
            origin = origin; direction = direction; 
        }
        __device__ Vector origin() const { return origin; }
        __device__ Vector direction() const { return direction; }
        __device__ Vector at(double t) const {
            return origin + t * direction;
        }
};

#endif // RAY_H
