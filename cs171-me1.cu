#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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

__global__
void kernel_1t1e(float *d_A, float *d_B, float *d_C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_A[idx] = d_B[idx] + d_C[idx];
    }
}

__global__
void kernel_1t1r(float *d_A, float *d_B, float *d_C,rows) {
    int i = 0;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for(i = 0;i<rows;i++){
        d_A[i*rows + j] = d_B[i*rows + j] + d_C[i*rows + j];
    }
}

__global__
void kernel_1t1c(float *d_A, float *d_B, float *d_C,rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 0;

    for(j = 0;j<rows;j++){
        d_A[i*rows + j] = d_B[i*rows + j] + d_C[i*rows + j];
    }
}

void hostFunction(float *A, float *B, float *C, int rows) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    int numBlocks = 1;
    dim3 threadsPerBlock(rows,1);

    cudaMalloc(&d_A, rows*rows*sizeof(float));
    cudaMalloc(&d_B, rows*rows*sizeof(float));
    cudaMalloc(&d_C, rows*rows*sizeof(float));

    // Copy values to device memory
    cudaMemcpy(d_B, B, rows*rows*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, rows*rows*sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel function
    int size = rows*rows;
    kernel_1t1e<<<(int) (rows/1024) + 1, 1024>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();

    // Get return value
    cudaMemcpy(A, d_A, rows*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // Call kernel function
    kernel_1t1r<<<threadsPerBlock, numBlocks>>>(d_A, d_B, d_C,rows);

    // Get return value
    cudaMemcpy(A, d_A, rows*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // Call kernel function
    kernel_1t1c<<<threadsPerBlock, numBlocks>>>(d_A, d_B, d_C,rows);

    // Get return value
    cudaMemcpy(A, d_A, rows*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /*
    int numBlocks = 1;
    dim3 threadsPerBlock(rows,1);
    kernel_1t1r<<<numBlocks,threadsPerBlock>>>
    */

    /*
    for (int i=0; i<rows; i++) {
        for (int j=0; j<rows; j++) {
            A[i*rows + j] = B[i*rows + j] + C[i*rows + j];
        }
    }
    */
}

int main() {
    // Device Query first
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printDevProp(deviceProp);
    }

    // In my (Francis) local machine there is only one CUDA machine, so I'll hardcode that one here
    // Allocate memory
    const int rows = 16;
    const int cols = rows;
    float *A, *B, *C;
    A = (float*) malloc(sizeof(float) * rows * cols);
    B = (float*) malloc(sizeof(float) * rows * cols);
    C = (float*) malloc(sizeof(float) * rows * cols);

    // Generate the values
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            B[i*rows + j] = (float) rand() / (float) (RAND_MAX / 100);
            C[i*rows + j] = (float) rand() / (float) (RAND_MAX / 100);
        }
    }

    // Call the host function
    hostFunction(A, B, C, rows);

    printf("A:\n");
    printMatrix(A, rows);
    printf("B:\n");
    printMatrix(B, rows);
    printf("C:\n");
    printMatrix(C, rows);

    // Free memory
    free(A);
    free(B);
    free(C);

    printf("Done!\n");
}