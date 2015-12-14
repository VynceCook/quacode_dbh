#ifdef QUACODE_USE_CUDA
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <cuda/kernels.hh>
#include <cuda/constraints.hh>
#include <cuda/cuda.hh>

__global__ void fooKernel() {
    printf("Foo on GPU.\n");
}

void foo() {
    printf("Before foo\n");
    fooKernel<<<1,1>>>();
    CCR(cudaGetLastError());
    CCR(cudaDeviceSynchronize());
    printf("After foo\n");
}

#endif
