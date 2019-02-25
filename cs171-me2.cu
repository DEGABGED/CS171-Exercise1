#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define TILE_WIDTH 16

void printDevProp(cudaDeviceProp devProp) {
    // Source: https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
    printf("%s\n", devProp.name);
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Total global memory:           %u bytes\n", devProp.totalGlobalMem);
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Total shared memory per block: %u\n",devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Total constant memory:         %u\n", devProp.totalConstMem);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    printf("Maximum threads per dimension: %d,%d,%d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    return;
}

void printMatrix(float *A, int rows, int cols) {
    printf("[\n");
    for (int i=0; i<rows; i++) {
        printf("  [");
        for (int j=0; j<cols; j++) {
            printf("%.2f, ", A[i*cols + j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

void printMatrixFlat(float *A, int rows, int cols) {
    printf("[");
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            printf("%.2f, ", A[i*cols + j]);
        }
    }
    printf("]\n");
}

void populateMatrix(float *A, int rows, int cols) {
    // Generate the values
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            A[i*cols + j] = (float) rand() / (float) (RAND_MAX / 100);
        }
    }
}

__global__
void matmul_rec_glob(float *d_A, float *d_B, float *d_C, int n, int m, int k) {
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if ((rows < n) && (cols < m)) {
        float val = 0;
        for (int i = 0; i < k; i++) {
            val += d_B[rows*k + i] * d_C[i*m + cols];
        }
        d_A[rows*m + cols] = val;
    }
}

__global__
void matmul_rec_shar(float *d_A, float *d_B, float *d_C, int n, int m, int k) {

    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float C_shared[TILE_WIDTH][TILE_WIDTH];

    int rows = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int cols = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if ((rows < n) && (cols < m)) {
        float val = 0;
        for (int i = 0; i<k/TILE_WIDTH; ++i){
            B_shared[threadIdx.y][threadIdx.x] = d_B[rows*k + i*TILE_WIDTH + threadIdx.x];
            C_shared[threadIdx.y][threadIdx.x] = d_C[(i*TILE_WIDTH + threadIdx.y)*k + cols];
            __syncthreads();
            for (int j = 0; j < TILE_WIDTH; ++j) {
                val += B_shared[threadIdx.y][j] * C_shared[j][threadIdx.x];
            }
            d_A[rows*k + cols] = val;
        }
    }
}

double hostFunction(float *A, float *B, float *C, int n, int m, int k, int blockSize, int kernel_choice) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    // A = B * C
    // A: n x m
    // B: n x k
    // C: k x m
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, n*m*sizeof(float));
    cudaMalloc(&d_B, n*k*sizeof(float));
    cudaMalloc(&d_C, k*m*sizeof(float));

    // Copy values to device memory
    cudaMemcpy(d_B, B, n*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, k*m*sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel function
    const int dimY = 16;
    const int dimX = 16;
    dim3 dimBlock(dimX, dimY, 1);
    dim3 dimGrid(ceil((float) n / dimX), ceil((float) m / dimY), 1);

    cudaEventRecord(start);
    if (kernel_choice == 0) {
        matmul_rec_glob<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n, m, k);
    }
    else if (kernel_choice == 1) {
        matmul_rec_shar<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n, m, k);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Get return value
    cudaMemcpy(A, d_A, n*m*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return milliseconds;
}

int main() {
    // Device Query first
    int deviceCount;
    int blockSize = 1024;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printDevProp(deviceProp);
        blockSize = deviceProp.maxThreadsPerBlock;
    }

    // In my (Francis) local machine there is only one CUDA machine, so I'll hardcode that one here
    // Allocate memory
    // A = B * C
    // A: n x m
    // B: n x k
    // C: k x m
    const int n = 16;
    const int m = 16;
    const int k = 16;
    float *A, *B, *C;
    A = (float*) malloc(sizeof(float) * n * m);
    B = (float*) malloc(sizeof(float) * n * k);
    C = (float*) malloc(sizeof(float) * k * m);

    // Benchmarking
    // Source: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    int kernel = 0;
    int runs = 5;
    double time_spent = 0.0;
    double ave_time = 0.0;
    printf("\n");

    while (kernel < 2) {
        printf("#%d:\t", kernel);
        for (int run=0; run<runs; run++) {
            populateMatrix(B, n, k);
            populateMatrix(C, k, m);
            time_spent = hostFunction(A, B, C, n, m, k, blockSize, kernel);
            ave_time += time_spent;
            printf("%.4f\t", time_spent);
        }
        ave_time /= runs;
        printf("Ave: %.4f\n", ave_time);
        kernel++;
    }

    // Check matrices
    printMatrix(A, n, m);
    printMatrix(B, n, k);
    printMatrix(C, k, m);

    // printMatrixFlat(A, n, m);
    // printMatrixFlat(B, n, k);
    // printMatrixFlat(C, k, m);

    // Free memory
    free(A);
    free(B);
    free(C);

    printf("\nDone!\n");
}