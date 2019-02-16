#include<stdio.h>
#include<cuda.h>

void printDevProp(cudaDeviceProp devProp) {
    printf("%s\n", devProp.name);
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Total global memory:           %u", devProp.totalGlobalMem);
    printf(" bytes\n");
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Total shared memory per block: %u\n",devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Total constant memory:         %u\n",   devProp.totalConstMem);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    printf("Maximum threads per dimension: %d,%d,%d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    return;
}

int main(void) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printDevProp(deviceProp);
    }
}