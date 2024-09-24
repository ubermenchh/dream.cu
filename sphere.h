#ifndef SPHERE_H 
#define SPHERE_H 

#include "hittable.h"

class Sphere: public Hittable {
    public: 
        __device__ Sphere() {}
        __device__ Sphere(Vector cent, double r, Material* m): center(cent), radius(r), mat_ptr(m) {};
        __device__ virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const;
        
        Vector center;
        double radius;
        Material* mat_ptr;
};
    
__device__ bool Sphere::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    Vector oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double b = dot(oc, r.direction());
    double c = dot(oc, oc) - radius * radius;
    double discriminant = b*b - a*c;
    
    if (discriminant > 0) {
        double temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif // SPHERE_H
