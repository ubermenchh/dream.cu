// Dream Ray-Tracing Framework in CUDA 

#include "dream.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error: %d at %s:%d '%s'\n", (int)result, file, line, func);
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vector_t color(Ray_t* ray, Hittable* world) {
    Hit_Record rec;
    if ((world)->hit(world, *ray, Interval(0.0f, FLT_MAX), &rec)) {
        return Vector(1, 0, 0);
    }
    Vector_t unit_direction = unit_vector(ray_direction(*ray));
    double t = 0.5f * (unit_direction.y + 1.0f);
    return vector_add(vector_scalar_mul(Vector(1.0, 1.0, 1.0), 1.f - t), 
                      vector_scalar_mul(Vector(0.5, 0.7, 1.0), t));
}

__global__ void render(Vector_t* fb, int max_x, int max_y, 
                       Vector_t lower_left_corner, Vector_t horizontal,
                       Vector_t vertical, Vector_t origin, Hittable* world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    double u = (double)(i / max_x);
    double v = (double)(j / max_y);
    Vector_t direction = vector_add(lower_left_corner, 
                                    vector_add(vector_scalar_mul(horizontal, u),
                                               vector_scalar_mul(vertical, v)));
    Ray_t ray = Ray(origin, direction);
    fb[pixel_index] = color(&ray, world);
}

__global__ void create_world(Hittable_List_t* d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Initializing World!\n");
        
        Sphere_t* sphere1 = Sphere(Vector(0, 0, -1), 0.5);
        Sphere_t* sphere2 = Sphere(Vector(0, -100.5, -1), 100);

        printf("Adding Sphere 1\n");
        hittable_list_add(d_world, (Hittable*)sphere1);
        printf("Adding Sphere 2\n");
        hittable_list_add(d_world, (Hittable*)sphere2);
        
        printf("World created with %lu objects.\n", d_world->size);
        printf("Sphere 1 with radius %f: ", sphere1->radius);
        vector_print(sphere1->center);
        printf("Sphere 2 with radius %f: ", sphere2->radius);
        vector_print(sphere2->center);
    }
}

__global__ void free_world(Hittable_List_t* d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (size_t i = 0; i < d_world->size; i++) {
            Sphere_t* sphere = (Sphere_t*)d_world->objects[i];
            free_sphere(sphere);
        }
        hittable_list_destroy(d_world);
    }
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

    Hittable_List_t* h_world = Hittable_List();
    Hittable_List_t* d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable_List_t)));
    checkCudaErrors(cudaMemcpy(d_world, h_world, sizeof(Hittable_List_t), cudaMemcpyHostToDevice));
    create_world <<< 1, 1 >>> (d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t start, stop;
    start = clock();
    
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1); // (151, 76)
    dim3 threads(tx, ty); // (8, 8)
    
    Vector_t lower_left_corner = Vector(-2.0, -1.0, -1.0);
    Vector_t horizontal = Vector(4.0, 0.0, 0.0);
    Vector_t vertical = Vector(0.0, 2.0, 0.0);
    Vector_t origin = Vector(0.0, 0.0, 0.0);
    
    render <<< blocks, threads >>> (fb, nx, ny, lower_left_corner, horizontal, vertical, origin, (Hittable*)d_world);
    
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
    
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<< 1, 1 >>> (d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    
    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    
    return 0;
}
