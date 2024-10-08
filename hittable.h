#ifndef HITTABLE_H 
#define HITTABLE_H 

class Material;

struct HitRecord {
    double t;
    Vector p;
    Vector normal;
    Material* mat_ptr;
};

class Hittable {
    public:
        __device__ virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
};

#endif // HITTABLE_H
