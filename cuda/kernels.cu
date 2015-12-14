#ifdef QUACODE_USE_CUDA
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <cuda/kernels.hh>
#include <cuda/constraints.hh>
#include <cuda/cuda.hh>

extern CUDA_DEVICE __constant__ uintptr_t cstrData[1024];

__global__ void fooKernel() {
    printf("Foo on GPU.\n");

    for (size_t i = 0; cstrData[8 * i] != CSTR_NO; ++i) {
        printf("Une contrainte\n");
    }
}

void foo() {
    printf("Before foo\n");
    fooKernel<<<1,1>>>();
    CCR(cudaGetLastError());
    CCR(cudaDeviceSynchronize());
    printf("After foo\n");
}

#endif
