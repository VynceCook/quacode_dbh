#ifndef __KERNELS_H_
#define __KERNELS_H_

#include <cuda/constraints.hh>
#include <cuda/cuda.hh>

#ifdef __CUDACC__
CUDA_GLOBAL void fooKernel();
#endif


void foo();
bool evaluateCstrs(Constraint **, size_t, const int *);

#endif
