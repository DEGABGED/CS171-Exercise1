#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

void printDevProp(cudaDeviceProp devProp) {
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

void printMatrix(float *A, int dim) {
    printf("[\n");
    for (int i=0; i<dim; i++) {
        printf("  [");
        for (int j=0; j<dim; j++) {
            printf("%.2f, ", A[i*dim + j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

void populateMatrix(float *A, int dim) {
    // Generate the values
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            A[i*dim + j] = (float) rand() / (float) (RAND_MAX / 100);
        }
    }
}

__global__
void kernel_1t1e(float *d_A, float *d_B, float *d_C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_A[idx] = d_B[idx] + d_C[idx];
    }
}

__global__
void kernel_1t1r(float *d_A, float *d_B, float *d_C, int rows) {
    int i = 0;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < rows) {
        for(i = 0;i<rows;i++){
            d_A[i*rows + j] = d_B[i*rows + j] + d_C[i*rows + j];
        }
    }
}

__global__
void kernel_1t1c(float *d_A, float *d_B, float *d_C, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 0;

    if (i < rows) {
        for(j = 0;j<rows;j++){
            d_A[i*rows + j] = d_B[i*rows + j] + d_C[i*rows + j];
        }
    }
}

double hostFunction(float *A, float *B, float *C, int rows, int blockSize, int kernel_choice) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, rows*rows*sizeof(float));
    cudaMalloc(&d_B, rows*rows*sizeof(float));
    cudaMalloc(&d_C, rows*rows*sizeof(float));

    // Copy values to device memory
    cudaMemcpy(d_B, B, rows*rows*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, rows*rows*sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel function
    int size = rows*rows;
    int numBlocks = (int) (rows/blockSize) + 1;
    dim3 threadsPerBlock(blockSize,1);

    cudaEventRecord(start);
    if (kernel_choice == 0) {
        int numBlocks = (int) (size/blockSize) + 1;
        kernel_1t1e<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, size);
    } else if (kernel_choice == 1) {
        kernel_1t1r<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows);
    } else if (kernel_choice == 2) {
        kernel_1t1c<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Get return value
    cudaMemcpy(A, d_A, rows*rows*sizeof(float), cudaMemcpyDeviceToHost);

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
    const int rows = 64;
    const int cols = rows;
    float *A, *B, *C;
    A = (float*) malloc(sizeof(float) * rows * cols);
    B = (float*) malloc(sizeof(float) * rows * cols);
    C = (float*) malloc(sizeof(float) * rows * cols);

    // Call the host function
    // Benchmarking
    int kernel = 0;
    int runs = 10;
    double time_spent = 0.0;
    double ave_time = 0.0;
    printf("\n");

    while (kernel < 3) {
        printf("#%d:\t", kernel);
        for (int run=0; run<runs; run++) {
            populateMatrix(B, rows);
            populateMatrix(C, rows);
            time_spent = hostFunction(A, B, C, rows, blockSize, kernel);
            ave_time += time_spent;
            printf("%.4f\t", time_spent);
        }
        ave_time /= runs;
        printf("Ave: %.4f\n", ave_time);
        kernel++;
    }

    // Free memory
    free(A);
    free(B);
    free(C);

    printf("\nDone!\n");
}