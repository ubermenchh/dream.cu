// Dream Ray-Tracing Framework in CUDA 

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dream.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result,const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error: %d at %s:%d '%s'\n", (int)result, file, line, func);
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(Vector_t* center, float radius, Ray_t* ray) {
    Vector_t oc = vector_sub(ray_origin(*ray), *center);
    double a = vector_dot(ray_direction(*ray), ray_direction(*ray));
    double b = 2.0f * vector_dot(oc, ray_direction(*ray));
    double c = vector_dot(oc, oc) - radius*radius;
    double discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0.0f);
}

__device__ Vector_t color(Ray_t* ray) {
    if (hit_sphere(&(Vector(0, 0, -1)), 0.5, ray))
        return Vector(1, 0, 0);
    Vector_t unit_direction = unit_vector(ray_direction(*ray));
    double t = 0.5f * (unit_direction.y + 1.0f);
    return vector_add(vector_scalar_mul(Vector(1.0, 1.0, 1.0), 1.f - t), 
                      vector_scalar_mul(Vector(0.5, 0.7, 1.0), t));
}

__global__ void render(Vector_t* fb, int max_x, int max_y, 
                       Vector_t lower_left_corner, Vector_t horizontal,
                       Vector_t vertical, Vector_t origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    double u = (double)i / (double)max_x;
    double v = (double)j / (double)max_y;
    Vector_t direction = vector_add(lower_left_corner, 
                                    vector_add(vector_scalar_mul(horizontal, u),
                                               vector_scalar_mul(vertical, v)));
    Ray_t ray = Ray(origin, direction);
    fb[pixel_index] = color(&ray);
}

int main(void) {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;
    
    fprintf(stderr, "Rendering a %dx%d image ", nx, ny);
    fprintf(stderr, "in %d x %d blocks.\n", tx, ty);
    
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Vector_t);
    
    // allocate FB
    Vector_t* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    
    clock_t start, stop;
    start = clock();
    
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1); // (151, 76)
    dim3 threads(tx, ty); // (8, 8)
    
    Vector_t lower_left_corner = Vector(-2.0, -1.0, -1.0);
    Vector_t horizontal = Vector(4.0, 0.0, 0.0);
    Vector_t vertical = Vector(0.0, 2.0, 0.0);
    Vector_t origin = Vector(0.0, 0.0, 0.0);
    
    render <<< blocks, threads >>> (fb, nx, ny, lower_left_corner, horizontal, vertical, origin);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double total_time = (double)(stop - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "took %f seconds.\n", total_time);
    
    // Output FB as image 
    printf("P3\n%d %d\n255\n", nx, ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            
            int ir = (int)(255.999 * fb[pixel_index].x);
            int ig = (int)(255.999 * fb[pixel_index].y);
            int ib = (int)(255.999 * fb[pixel_index].z);
            
            printf("%d %d %d\n", ir, ig, ib);
        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}
