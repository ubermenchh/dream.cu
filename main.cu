// Dream Ray-Tracing Framework in CUDA 

#include <cuda.h>
#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include "vector.h"
#include "ray.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* file, int const line) {
    if (result) {
        std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vector color(const Ray& r, Hittable** world, curandState* local_rand_state) {
    Ray cur_ray = r;
    Vector cur_attenuation = Vector(1.0, 1.0, 1.0); 
    
    for (int i = 0; i < 50; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vector attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return Vector(0.0, 0.0, 0.0);
            }
        } else {   
            Vector unit_direction = unit_vector(cur_ray.direction());
            double t = 0.5f * (unit_direction.y() + 1.0f);
            Vector c = (1.0f - t) * Vector(1.0, 1.0, 1.0) + t * Vector(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vector(0.0, 0.0, 0.0);
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    } 
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    
    // Each thread gets the same seed, different sequence number, no offset
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(Vector* fb, int max_x, int max_y, int ns, Camera** cam, Hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vector col = Vector(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        double u = double(i + curand_uniform(&local_rand_state)) / double(max_x);
        double v = double(j + curand_uniform(&local_rand_state)) / double(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= double(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new Sphere(Vector(0, -1000.0, -1), 1000, new Lambertian(Vector(0.5, 0.5, 0.5)));
        
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                double choose_mat = RND;
                Vector center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(center, 0.2, new Lambertian(Vector(RND*RND, RND*RND, RND*RND)));
                } else if (choose_mat < 0.95f) {
                    d_list[i++] = new Sphere(center, 0.2, new Metal(Vector(0.5f * (1.f + RND), 0.5f * (1.f + RND), 0.5f * (1.f + RND)), 0.5f * RND));
                } else {
                    d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
        d_list[i++] = new Sphere(Vector(0, 1, 0), 1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(Vector(-4, 1, 0), 1.0, new Lambertian(Vector(0.4, 0.2, 0.1)));
        d_list[i++] = new Sphere(Vector(4, 1, 0), 1.0, new Metal(Vector(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new HittableList(d_list, 22*22+1+3);
        
        Vector look_from(13, 2, 3);
        Vector look_at(0, 0, 0);
        double dist_to_focus = 10;
        double aperture = 0.1; 
        *d_camera = new Camera(look_from,
                               look_at,
                               Vector(0, 1, 0), 
                               30.0, 
                               double(nx) / double(ny), 
                               aperture,
                               dist_to_focus);
    }
}

__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    for (int i = 0; i < 22*22+1+3; i++) {
        delete ((Sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200, ny = 800, ns = 100;
    int tx = 8, ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Vector);

    // Allocate FB 
    Vector* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));
    
    // 2nd random state for world creation
    rand_init <<< 1, 1 >>> (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // create world
    Hittable** d_list;
    int num_hittables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(Hittable*)));
    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    create_world <<< 1, 1 >>> (d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t start, stop;
    start = clock();

    // Render our buffer 
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty); 
    render_init <<< blocks, threads >>> (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render <<< blocks, threads >>> (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
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
    free_world <<< 1, 1 >>> (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    
    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
