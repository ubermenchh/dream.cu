#ifndef DREAM_H
#define DREAM_H 

#include <math.h> 
#include <stdio.h> 
#include <stdbool.h> 
#include <stdlib.h>

// Constants 
#define PI (3.1415926535897932385)

// Utility Functions 
__host__ __device__ static inline double degrees_to_radians(double degrees) {
    return degrees * PI / 180.0;
}

__host__ __device__ static inline double rand_double() {
    return rand() / (RAND_MAX + 1.0);
}

__host__ __device__ static inline double random_double(double min, double max) {
    return min + (max - min) * rand_double();
} 

// Common Headers 
#include "vector.h"

#endif // DREAM_H
