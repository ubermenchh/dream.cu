#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H 

#include "hittable.h" 
#include <stdbool.h>
#include <stdlib.h>

#define MAX_HITTABLES 1024

typedef struct {
    Hittable base;
    Hittable* objects[MAX_HITTABLES];
    size_t size;
} Hittable_List_t;

__device__ static inline void hittable_list_destroy(Hittable_List_t* list) {
    if (list != NULL) {
        free(list->objects);
        free(list);
    }
}

__device__ static inline void hittable_list_clear(Hittable_List_t* list) {
    for (size_t i = 0; i < list->size; i++) {
        free(list->objects[i]);
    }
    free(list->objects);
    list->size = 0;
}

__device__ static inline void hittable_list_add(Hittable_List_t* list, Hittable* object) {
    if (list->size < MAX_HITTABLES) {
        list->objects[list->size++] = object;
    }
}

__device__ static inline bool hittable_list_hit(Hittable* hittable, Ray_t ray, Interval_t ray_t, Hit_Record* rec) {
    Hittable_List_t* list = (Hittable_List_t*)hittable;
    Hit_Record temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_t.max;

    for (size_t i = 0; i < list->size; i++) {
        if (list->objects[i]->hit(list->objects[i], ray, Interval(ray_t.min, closest_so_far), &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
        }
    }
    return hit_anything;
}

__host__ __device__ static inline Hittable_List_t* Hittable_List() {
    Hittable_List_t* list = (Hittable_List_t*)malloc(sizeof(Hittable_List_t));; 
    if (list != NULL) {
        list->base.hit = hittable_list_hit;
        list->size = 0;
    }
    return list;
} 

#endif // HITTABLE_LIST_H
