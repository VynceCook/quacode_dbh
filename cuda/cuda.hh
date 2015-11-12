#ifndef __CUDA_H_
#define __CUDA_H_

#include <stdio.h>


#ifndef __CUDACC__
#define CUDA_GLOBAL // You don't need to see this
#define CUDA_HOST
#define CUDA_DEVICE // You don't need to see this
#else

#define CUDA_GLOBAL __global__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CCR(__value) {								                           \
	cudaError_t _m_cudaStat = __value;							               \
	if (_m_cudaStat != cudaSuccess) {							               \
		fprintf(stderr, "Error %s at line %d in file %s\n",				       \
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		   \
		exit(1);									                           \
	} }

#endif
#endif
