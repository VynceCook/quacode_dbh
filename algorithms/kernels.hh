#ifndef __KERNELS_H_
#define __KERNELS_H_

#include <algorithms/constraints.hh>
#include <algorithms/cuda.hh>

#ifdef __CUDACC__
CUDA_GLOBAL void CstrEqKernel(CstrEq **, size_t v0, int v2);
CUDA_GLOBAL void CstrBoolKernel(CstrBool** ptr, bool p0, size_t v0, TOperatorType op, bool p1, size_t v1, TComparisonType cmp, size_t v2);
CUDA_GLOBAL void CstrPlusKernel(CstrPlus **ptr, int n0, size_t v0, int n1, size_t v1, TComparisonType cmp, size_t v2);
CUDA_GLOBAL void CstrTimesKernel(CstrTimes **ptr, int n, size_t v0, size_t v1, TComparisonType cmp, size_t v2);
CUDA_GLOBAL void CstrLinearKernel(CstrLinear **ptr, size_t * poly, size_t size, TComparisonType cmp, size_t v0);
CUDA_GLOBAL void fooKernel();
#endif


void foo();
bool evaluateCstrs(Constraint **, size_t, const int *);

#endif
