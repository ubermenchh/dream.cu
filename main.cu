// Dream Ray-Tracing Framework in CUDA 

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result,const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error: %d at %s:%d '%s'\n", (int)result, file, line, func);
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = (float)i / max_x;
    fb[pixel_index + 1] = (float)j / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main(void) {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;
    
    fprintf(stderr, "Rendering a %dx%d image ", nx, ny);
    fprintf(stderr, "in %d x %d blocks.\n", tx, ty);
    
    int num_pixels = nx * ny;
    size_t fb_size = 3 * num_pixels * sizeof(float);
    
    // allocate FB
    float *fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    
    clock_t start, stop;
    start = clock();
    
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<< blocks, threads >>> (fb, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double total_time = (double)(stop - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "took %f seconds.\n", total_time);
    
    // Output FB as image 
    printf("P3\n%d %d\n255\n", nx, ny);
    for (int j = 0; j <ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * 3 * nx + i * 3;
            
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            
            int ir = (int)(255.999 * r);
            int ig = (int)(255.999 * g);
            int ib = (int)(255.999 * b);
            
            printf("%d %d %d\n", ir, ig, ib);
        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}
