#include <cuda/helper.hh>

size_t * pushPolyToGPU(size_t * poly, size_t size) {
    size_t * ret;

    CCR(cudaMalloc((void**)&ret, size * sizeof(size_t)));
    CCR(cudaMemcpy(ret, poly, size * sizeof(size_t), cudaMemcpyHostToDevice));

    return ret;
}
