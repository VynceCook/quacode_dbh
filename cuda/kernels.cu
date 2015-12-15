#ifdef QUACODE_USE_CUDA
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <cuda/kernels.hh>
#include <cuda/constraints.hh>
#include <cuda/cuda.hh>

extern CUDA_DEVICE __constant__ uintptr_t   cstrData[1024];
extern CUDA_DEVICE __constant__ TVarType    cstrType[CSTR_MAX_VAR];
extern CUDA_DEVICE __constant__ Gecode::TQuantifier cstrQuan[CSTR_MAX_VAR];
extern CUDA_DEVICE __constant__ int         cstrDom[CSTR_MAX_VAR];

__global__ void fooKernel() {
    printf("Foo on GPU.\n");

    for (size_t i = 0; cstrData[8 * i] != CSTR_NO; ++i) {
        printf("Une contrainte\n");
    }

    for (size_t i = 0; i < 32; ++i) {
        printf("Var %lu : %d, %d, {%d, %d}\n", i, cstrType[i], cstrQuan[i], cstrDom[2 * i], cstrDom[2 * i + 1]);
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
