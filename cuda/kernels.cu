#ifdef QUACODE_USE_CUDA
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <cuda/kernels.hh>
#include <cuda/constraints.hh>
#include <cuda/cuda.hh>

__device__ Constraint* dCstrs[512];
__device__ size_t      dNbCstrs;

__global__ void fooKernel() {
    printf("Foo on GPU.\n");

    for (size_t i = 0; i < dNbCstrs; ++i) {
        dCstrs[i]->describe();
    }

}

void foo() {
    printf("Before foo\n");
    fooKernel<<<1,1>>>();
    CCR(cudaGetLastError());
    CCR(cudaDeviceSynchronize());
    printf("After foo\n");
}

bool evaluateCstrs(Constraint ** cstrs, size_t nbCstrs, const int * candidat) {
    return true;
}

#endif
