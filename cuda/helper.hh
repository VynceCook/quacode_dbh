#ifndef __HELPER_H_
#define __HELPER_H_

#include <cuda/cuda.hh>

CUDA_DEVICE CUDA_HOST unsigned int int2uint(int i);
CUDA_DEVICE CUDA_HOST int     uint2int(unsigned int i);

#endif
