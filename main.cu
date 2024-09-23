// Dream Ray-Tracing Framework in CUDA 

#include <cuda.h>
#include <iostream>
#include <time.h>
#include <float.h>

#include "vector.h"
#include "ray.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result,const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vector color(const Ray& r, Hittable** world) {
    HitRecord rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) 
        return 0.5f * Vector(rec.normal.x() + 1.f, rec.normal.y() + 1.f, rec.normal.z() + 1.f);
    Vector unit_direction = unit_vector(r.direction());
    double t = 0.5f * (unit_direction.y() + 1.f);
    return (1.0f - t) * Vector(1.0, 1.0, 1.0) + t * Vector(0.5, 0.7, 1.0);
}

__global__ void render(Vector* fb, int max_x, int max_y, Vector lower_left_corner, 
                       Vector horizontal, Vector vertical, Vector origin, Hittable** world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    double u = double(i) / double(max_x);
    double v = double(j) / double(max_y);
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r, world);
}

__global__ void create_world(Hittable** d_list, Hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)     = new Sphere(Vector(0, 0, -1), 0.5);
        *(d_list + 1) = new Sphere(Vector(0, -100.5, -1), 100);
        *d_world      = new HittableList(d_list, 2);
    }
}

__global__ void free_world(Hittable** d_list, Hittable** d_world) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
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

    // create world
    Hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));
    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    create_world <<< 1, 1 >>> (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t start, stop;
    start = clock();

    // Render our buffer 
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty); 
    render <<< blocks, threads >>> (fb, nx, ny, 
                                    Vector(-2.0, -1.0, -1.0),
                                    Vector(4.0, 0.0, 0.0),
                                    Vector(0.0, 2.0, 0.0),
                                    Vector(0.0, 0.0, 0.0),
                                    d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double total_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << total_time << " seconds.\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx ; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.999 * fb[pixel_index].x());
            int ig = int(255.999 * fb[pixel_index].y());
            int ib = int(255.999 * fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<< 1, 1 >>> (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    
    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
