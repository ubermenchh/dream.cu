// Dream Ray-Tracing Framework in CUDA 

#include <cuda.h>
#include <iostream>
#include <time.h>

#include "vector.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result,const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vector color(const Ray& r) {
    Vector unit_direction = unit_vector(r.direction());
    double t = 0.5f * (unit_direction.y() + 1.f);
    return (1.f - t) * Vector(1.0, 1.0, 1.0) + t * Vector(0.5, 0.7, 1.0);
}

__global__ void render(Vector* fb, int max_x, int max_y, Vector lower_left_corner, 
                       Vector horizontal, Vector vertical, Vector origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    double u = double(i) / double(max_x);
    double v = double(j) / double(max_y);
    Ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(r);
}

int main() {
    int nx = 1200, ny = 600;
    int tx = 8, ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Vector);

    // Allocate FB 
    Vector* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();

    // Render our buffer 
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty); 
    render <<< blocks, threads >>> (fb, nx, ny, 
                                    Vector(-2.0, -1.0, -1.0),
                                    Vector(4.0, 0.0, 0.0),
                                    Vector(0.0, 2.0, 0.0),
                                    Vector(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double total_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << total_seconds << " seconds.\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = 0; i < nx; i++) {
        for (int j = ny-1; j >= 0; j--) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.999 * fb[pixel_index].x());
            int ig = int(255.999 * fb[pixel_index].y());
            int ib = int(255.999 * fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
}
