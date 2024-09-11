#ifndef VECTOR_H
#define VECTOR_H 

typedef struct Vector_t {
    double x;
    double y;
    double z;
} Vector_t;

typedef Vector_t Point_t;

__host__ __device__ static inline Vector_t Vector(double x, double y, double z) {
    return (Vector_t){.x = x, .y = y, .z = z};
}  

static inline void vector_print(Vector_t v) {
    printf("Vector(x=%f, y=%f, z=%f)\n", v.x, v.y, v.z);
} 

__host__ __device__ static inline Vector_t vector_negate(Vector_t v) {
    Vector_t out = Vector(-v.x, -v.y, -v.z);
    return out;
}

__host__ __device__ static inline Vector_t vector_add(Vector_t v, Vector_t w) {
    Vector_t out = Vector(v.x + w.x, v.y + w.y, v.z + w.z);
    return out;
}

__host__ __device__ static inline Vector_t vector_add_(Vector_t* v, Vector_t w) {
    v->x += w.x; 
    v->y += w.y; 
    v->z += w.z;
    return *v;
}

__host__ __device__ static inline Vector_t vector_sub(Vector_t v, Vector_t w) {
    Vector_t out = Vector(v.x - w.x, v.y - w.y, v.z - w.z);
    return out;
}

__host__ __device__ static inline Vector_t vector_sub_(Vector_t* v, Vector_t w) {
    v->x -= w.x; 
    v->y -= w.y; 
    v->z -= w.z;
    return *v;
}

__host__ __device__ static inline Vector_t vector_mul(Vector_t v, Vector_t w) {
    Vector_t out = Vector(v.x * w.x, v.y * w.y, v.z * w.z);
    return out;
}

__host__ __device__ static inline Vector_t vector_mul_(Vector_t* v, Vector_t w) {
    v->x += w.x; 
    v->y += w.y; 
    v->z += w.z;
    return *v;
}

__host__ __device__ static inline Vector_t _vector_scalar_mul(Vector_t v, double t) {
    v.x *= t;
    v.y *= t;
    v.z *= t;
    return v;
}

__host__ __device__ static inline Vector_t vector_scalar_mul(Vector_t v, double t) {
    Vector_t out = Vector(v.x * t, v.y * t, v.z * t);
    return out;
}

__host__ __device__ static inline Vector_t vector_scalar_div(Vector_t v, double t) {
    Vector_t out = Vector(v.x / t, v.y / t, v.z / t);
    return out;
}

__host__ __device__ static inline Vector_t vector_scalar_div_(Vector_t* v, double t) {
    v->x /= t; 
    v->y /= t; 
    v->z /= t;
    return *v;
}

__host__ __device__ static inline double vector_length_sq(Vector_t v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ static inline double vector_length(Vector_t v) {
    return sqrt(vector_length_sq(v));
}

__host__ __device__ static inline double vector_dot(Vector_t v, Vector_t w) {
    return (v.x * w.x) + (v.y * w.y) + (v.z * w.z);
}

__host__ __device__ static inline Vector_t vector_cross(Vector_t v, Vector_t w) {
    return Vector(v.y * w.z - v.z * w.y,
                  v.z * w.x - v.x * w.z,
                  v.x * w.y - v.y * w.x);
}

__host__ __device__ static inline Vector_t unit_vector(Vector_t v) {
    return vector_scalar_div(v, vector_length(v));
}

__host__ __device__ static inline Vector_t rand_vector() {
    return Vector(rand_double(), rand_double(), rand_double());
}

__host__ __device__ static inline Vector_t random_vector(double min, double max) {
    return Vector(random_double(min, max), random_double(min, max), random_double(min, max));
}

__host__ __device__ static inline Vector_t random_vector_in_unit_sphere() {
    while (true) {
        Vector_t p = random_vector(-1, 1);
        if (vector_length(p) < 1) 
            return p;
    }
}

__host__ __device__ static inline Vector_t random_unit_vector() {
    return unit_vector(random_vector_in_unit_sphere());
}

__host__ __device__ static inline Vector_t random_vector_on_hemisphere(Vector_t* normal) {
    Vector_t on_unit_sphere = random_unit_vector();
    if (vector_dot(on_unit_sphere, *normal) > 0.0) // in the same hemisphere as the normal 
        return on_unit_sphere;
    else 
        return vector_negate(on_unit_sphere);
}

__host__ __device__ static inline bool vector_near_zero(Vector_t vec) {
    // return true if the vector is close to zero in all directions 
    double s = 1e-8;
    return (fabs(vec.x) < s) && (fabs(vec.y) < s) && (fabs(vec.z) < s);
}

__host__ __device__ static inline Vector_t vector_reflect(Vector_t v, Vector_t n) {
    return vector_sub(v, vector_scalar_mul(n, 2*vector_dot(v, n)));
}

__host__ __device__ static inline Vector_t vector_refract(Vector_t uv, Vector_t n, double etai_over_etat) {
    double cos_theta = fmin(vector_dot(vector_negate(uv), n), 1.0);
    Vector_t r_out_perp = vector_scalar_mul(vector_add(uv, vector_scalar_mul(n, cos_theta)), etai_over_etat);
    Vector_t r_out_parallel = vector_scalar_mul(n, 
                                                 -sqrt(fabs(1.0 - vector_length_sq(r_out_perp))));
    return vector_add(r_out_perp, r_out_parallel);
}

__host__ __device__ static inline Vector_t random_vector_in_unit_disk(void) {
    while (true) {
        Vector_t p = Vector(random_double(-1, 1), random_double(-1, 1), 0);
        if (vector_length_sq(p) < 1) 
            return p;
    }
}

#endif // VECTOR_H
