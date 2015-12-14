#ifndef __HELPER_H_
#define __HELPER_H_

#include <cuda/constraints.hh>

size_t * pushPolyToGPU(size_t * poly, size_t size);
unsigned int int2uint(int i);
int     uint2int(unsigned int i);

#endif
