#include <algorithms/kernels.hh>
#include <stdio.h>
#include <unistd.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CCR(__value) {								                           \
	cudaError_t _m_cudaStat = __value;							               \
	if (_m_cudaStat != cudaSuccess) {							               \
		fprintf(stderr, "Error %s at line %d in file %s\n",				       \
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		   \
		exit(1);									    \
	} }

__global__ void fooKernel() {
    printf("Foo on GPU.\n");
}

void foo() {
    printf("Before foo\n");
    fooKernel<<<1,1>>>();
    CCR(cudaGetLastError());
    CCR(cudaDeviceSynchronize());
    printf("After foo\n");
}
