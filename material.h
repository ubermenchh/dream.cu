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

__device__ double schlick(double cosine, double ref_idx) {
    double r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.f - r0) * pow((1.f - cosine), 5.f);
}

__device__ bool refract(const Vector& v, const Vector& n, double ni_over_nt, Vector& refracted) {
    Vector uv = unit_vector(v);
    double dt = dot(uv, n);
    double discriminant = 1.f - ni_over_nt * ni_over_nt * (1 - dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else {
        return false;
    }
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

class Dielectric: public Material {
    public:
        double ref_idx;
    
        __device__ Dielectric(double ri): ref_idx(ri) {}
        __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector& attenuation, Ray& scattered, curandState* local_rand_state) const {
            Vector outward_normal;
            Vector reflected = reflect(r_in.direction(), rec.normal);
            double ni_over_nt;
            attenuation = Vector(1.0, 1.0, 1.0);
            Vector refracted;
            double reflect_prob, cosine;
            
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine*cosine));
            } else {
                outward_normal = rec.normal;
                ni_over_nt = 1.f / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            } 
            
            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) { 
                reflect_prob = schlick(cosine, ref_idx);
            } else {
                reflect_prob = 1.f;
            } 
            
            if (curand_uniform(local_rand_state) < reflect_prob) {
                scattered = Ray(rec.p, reflected);
            } else { 
                scattered = Ray(rec.p, refracted);
            }
            return true;
        }
};

#endif // MATERIAL_H
