#include <cuda/helper.hh>

typedef union {
    int a;
    unsigned int b;
} intuint_u;

CUDA_DEVICE CUDA_HOST unsigned int int2uint(int i) {
    intuint_u tmp;
    tmp.a = i;

    return tmp.b;
}

CUDA_DEVICE CUDA_HOST int uint2int(unsigned int i) {
    intuint_u tmp;
    tmp.b = i;
    return tmp.a;
}
