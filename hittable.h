#ifndef HITTABLE_H 
#define HITTABLE_H 

#include "interval.h"

//typedef struct Material_t Material_t;

typedef struct Hit_Record {
    Vector_t p;
    Vector_t normal;
//    Material_t* mat;
    double t;
//    bool front_face;
} Hit_Record;

typedef struct Hittable Hittable;

struct Hittable {
    bool (*hit)(Hittable* self, Ray_t ray, Interval_t ray_t, Hit_Record* rec);
};

//__device__ static inline void set_face_normal(Hit_Record* hr, Ray_t ray, Vector_t outward_normal) {
//     // sets the hit record normal vector.
//    // NOTE: `outward_normal` is assumed to have unit length.
//    bool front_face = vector_dot(ray.direction, outward_normal) < 0;
//    hr->normal = front_face ? outward_normal : vector_negate(outward_normal);
//}

#endif // HITTABLE_H
