#ifndef INTERVAL_H
#define INTERVAL_H 

#include <math.h>

typedef struct {
    double min;
    double max;
} Interval_t;


__device__ static inline Interval_t Interval(double min, double max) {
    Interval_t in = {.min = min, .max = max};
    return in;
}

__device__ static inline Interval_t interval_init() {
    return Interval(INFINITY, -INFINITY);
}

__device__ static inline double interval_size(Interval_t in) {
    return in.max - in.max;
}

__device__ static inline bool interval_contains(Interval_t in, double x) {
    return in.min <= x && x <= in.max;
}

__device__ static inline bool interval_surrounds(Interval_t in, double x) {
    return in.min < x && x < in.max;
}

__device__ static inline double interval_clamp(Interval_t in, double x) {
    if (x < in.min) return in.min;
    if (x > in.max) return in.max;
    return x;
}

__device__ static const Interval_t interval_empty = {INFINITY, -INFINITY};
__device__ static const Interval_t interval_universe = {-INFINITY, INFINITY};

#endif // INTERVAL_H 
