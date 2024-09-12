#ifndef SPHERE_H 
#define SPHERE_H 

#include "hittable.h"
#include "interval.h"

typedef struct Sphere_t {
    Hittable base;

    Vector_t center;
    double radius;
    //Material_t* mat;
} Sphere_t;


__device__ static inline bool hit_sphere(Hittable* hittable, Ray_t ray, Interval_t ray_t, Hit_Record* rec) {
    Sphere_t* sphere = (Sphere_t*)hittable;
    Vector_t oc = vector_sub(sphere->center, ray.origin);
    double a = vector_length_sq(ray.direction);
    double h = vector_dot(ray.direction, oc);
    double c = vector_length_sq(oc) - (sphere->radius * sphere->radius);

    double discriminant = h*h - a*c;
    if (discriminant < 0) 
        return false;

    double sqrt_d = sqrt(discriminant);
    
    // Find the nearest root that lies in the acceptable range.
    double root = (h - sqrt_d) / a;
    if (!interval_surrounds(ray_t, root)) {
        root = (h + sqrt_d) / a;
        if (!interval_surrounds(ray_t, root)) 
            return false;
    }
    rec->t = root;
    rec->p = ray_at(ray, rec->t);
    //Vector_t outward_normal = vector_scalar_div(vector_sub(rec->p, sphere->center), sphere->radius); 
    //set_face_normal(rec, ray, outward_normal);
    //rec->mat = sphere->mat;

    return true;
}

__host__ __device__ static void init_sphere(Sphere_t* sphere, Vector_t center, float radius) {
    if (sphere) {
        sphere->base.hit = hit_sphere;
        sphere->center = center;
        sphere->radius = fmaxf(0.0f, radius);
    }
}

__host__ __device__ static inline Sphere_t* Sphere(Vector_t center, double radius) {
    Sphere_t* sphere;
    cudaError_t err = cudaMalloc((void**)&sphere, sizeof(Sphere_t));
    
    if (err != cudaSuccess) 
        return NULL;
    
    if (sphere != NULL) {
        sphere->base.hit = hit_sphere;
        sphere->center = center;
        sphere->radius = fmaxf(0.0f, radius);
    }
    return sphere;
}

__host__ __device__ static inline void free_sphere(Sphere_t* sphere) {
    cudaFree(sphere);
}

#endif // SPHERE_H
