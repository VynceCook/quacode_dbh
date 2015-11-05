#ifndef __KERNELS_H_
#define __KERNELS_H_

#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_GLOBAL
#define CUDA_HOST
#define CUDA_DEVICE
#endif

#include <algorithms/constraints.hh>

void foo();

#endif
