#ifdef QUACODE_USE_CUDA
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <algorithms/kernels.hh>
#include <algorithms/constraints.hh>
#include <algorithms/cuda.hh>

__device__ Constraint* dCstrs[512];
__device__ size_t      dNbCstrs;

__global__ void CstrEqKernel(CstrEq ** ptr, size_t v0, int v2) {
    CstrEq * ret = new CstrEq(v0, v2);

    dCstrs[dNbCstrs++] = ret;
    *ptr = ret;
}

__global__ void CstrBoolKernel(CstrBool** ptr, bool p0, size_t v0, TOperatorType op, bool p1, size_t v1, TComparisonType cmp, size_t v2) {
    CstrBool * ret = new CstrBool(p0, v0, CstrBool::getOpPtr(op), p1, v1, Constraint::getCmpPtr(cmp), v2);

    dCstrs[dNbCstrs++] = ret;
    *ptr = ret;
}

__global__ void CstrPlusKernel(CstrPlus **ptr, int n0, size_t v0, int n1, size_t v1, TComparisonType cmp, size_t v2) {
    CstrPlus * ret = new CstrPlus(n0, v0, n1, v1, Constraint::getCmpPtr(cmp), v2);

    dCstrs[dNbCstrs++] = ret;
    *ptr = ret;
}

__global__ void CstrTimesKernel(CstrTimes **ptr, int n, size_t v0, size_t v1, TComparisonType cmp, size_t v2) {
    CstrTimes * ret = new CstrTimes(n, v0, v1, Constraint::getCmpPtr(cmp), v2);

    dCstrs[dNbCstrs++] = ret;
    *ptr = ret;
}
__global__ void CstrLinearKernel(CstrLinear **ptr, size_t * poly, size_t size, TComparisonType cmp, size_t v0) {
    CstrLinear * ret = new CstrLinear(poly, size, Constraint::getCmpPtr(cmp), v0);

    dCstrs[dNbCstrs++] = ret;
    *ptr = ret;
}

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
