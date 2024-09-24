#ifndef MATERIAL_H
#define MATERIAL_H

struct HitRecord;

#include "ray.h"
#include "hittable.h"

#define RAND_VECTOR Vector(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ Vector random_in_unit_sphere(curandState* local_rand_state) {
    Vector p;
    do {
        p = 2.f * RAND_VECTOR - Vector(1, 1, 1);
    } while (p.length_squared() >= 1.f);
    return p;
}

__device__ Vector reflect(const Vector& v, const Vector& n) {
    return v - 2.f * dot(v, n) * n;
}

class Material {
    public:
        __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian: public Material {
    public:
        Vector albedo;
    
        __device__ Lambertian(const Vector& a): albedo(a) {}
        __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector& attenuation, Ray& scattered, curandState* local_rand_state) const {
            Vector target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo;
            return true;
        }
};

class Metal: public Material {
    public:
        Vector albedo;
        double fuzz;
    
        __device__ Metal(const Vector& a, double f): albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector& attenuation, Ray& scattered, curandState* local_rand_state) const {
            Vector reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
};

#endif // MATERIAL_H
