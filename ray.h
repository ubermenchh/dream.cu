#ifndef RAY_H 
#define RAY_H
 
#include "vector.h" 

typedef struct {
    Vector_t origin;
    Vector_t direction;
} Ray_t;

/* Initialize a ray */
__device__ static inline Ray_t Ray(Vector_t origin, Vector_t direction) {
    Ray_t ray = {.origin = origin, .direction = direction};
    return ray;
}

/* functions to get the origin and direction on the GPU */
__device__ static inline Vector_t ray_origin(Ray_t ray) {
    return ray.origin;
}
__device__ static inline Vector_t ray_direction(Ray_t ray) {
    return ray.direction;
}

/* Calculate the 3D position along the line. */
__device__ static inline Vector_t ray_at(Ray_t ray, double t) {
    // P_t = A + tb
    return vector_add(ray.origin, vector_scalar_mul(ray.direction, t));
}

#endif // RAY_H
