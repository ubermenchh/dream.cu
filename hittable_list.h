#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H 

#include "hittable.h"

class HittableList: public Hittable {
    public: 
        __device__ HittableList() {}
        __device__ HittableList(Hittable** l, int n) {
            list = l; list_size = n;
        }
        __device__ virtual bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const;
    
        Hittable** list;
        int list_size;
};
    
__device__ bool HittableList::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t; 
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif // HITTABLE_LIST_H
