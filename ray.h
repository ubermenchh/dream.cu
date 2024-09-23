#ifndef RAY_H 
#define RAY_H
 
#include "vector.h" 

class Ray {
    public:
        Vector orig;
        Vector dir;
    
        __device__ Ray() {}
        __device__ Ray(const Vector& orig, const Vector& dir): orig(orig), dir(dir) {}
        __device__ Vector origin() const { return orig; }
        __device__ Vector direction() const { return dir; }
        __device__ Vector at(double t) const {
            return orig + t * dir;
        }
};

#endif // RAY_H
