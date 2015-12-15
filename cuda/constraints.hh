#ifndef __CONSTRAINTS_H_
#define __CONSTRAINTS_H_

#include <quacode/quacodetypes.hh>
#include <cuda/cuda.hh>
#include <cstddef>
#include <stdint.h>

typedef bool (*cstrFuncPtr)(uintptr_t *, int *);

#define CSTR_NO         INTMAX_MAX
#define CSTR_MAX_VAR    512
#define CSTR_MAX_CSTR   128

#define CSTR_EQ_IDX     0x0
#define CSTR_AND_IDX    0x1
#define CSTR_OR_IDX     0x2
#define CSTR_IMP_IDX    0x3
#define CSTR_XOR_IDX    0x4
#define CSTR_PLUS_IDX   0x5
#define CSTR_TIMES_IDX  0x6
#define CSTR_LINEAR_IDX 0x7

#define opAnd(__p0, __v0, __p1, __v1)                                           \
            (((__p0) ? (__v0) : !(__v0)) && ((__p1) ? (__v1) : !(__v1)))
#define opOr(__p0, __v0, __p1, __v1)                                            \
            (((__p0) ? (__v0) : !(__v0)) || ((__p1) ? (__v1) : !(__v1)))
#define opImp(__p0, __v0, __p1, __v1)                                           \
            (!(((__p0) ? (__v0) : !(__v0)) && !((__p1) ? (__v1) : !(__v1))))
#define opXor(__p0, __v0, __p1, __v1)                                           \
            (!((__p0) ? (__v0) : !(__v0)) != !((__p1) ? (__v1) : !(__v1)))
#define opPlus(__n0, __v0, __n1, __v1)                                          \
            ((__n0) * (__v0) + (__n1) * (__v1))
#define opTimes(__n, __v0, __v1)                                                \
            ((__n) * (__v0) * (__v1))
#define opLinear(__v, __size, __sum)                                            \
            do {                                                                \
                for (int i = 0; i < __size; i += 2) {                           \
                    __sum += __v[i] * __v[i + 1];                               \
                }                                                               \
            } while(0)

CUDA_HOST   void pushVarToGPU(TVarType * type, Gecode::TQuantifier * quant, size_t size);
CUDA_HOST   void pushDomToGPU(int * dom, size_t size);
CUDA_HOST   void pushCstrToGPU(uintptr_t * cstrs, size_t size);

CUDA_DEVICE bool cstrValidate(int * c);

CUDA_DEVICE bool cstrEq(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrAndEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrAndNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrAndGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrAndGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrAndLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrAndLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrOrEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrOrNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrOrGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrOrGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrOrLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrOrLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrImpEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrImpNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrImpGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrImpGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrImpLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrImpLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrXorEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrXorNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrXorGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrXorGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrXorLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrXorLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrPlusEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrPlusNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrPlusGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrPlusGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrPlusLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrPlusLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrTimesEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrTimesNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrTimesGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrTimesGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrTimesLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrTimesLE(uintptr_t * data, int * c);

CUDA_DEVICE bool cstrLinearEQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrLinearNQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrLinearGQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrLinearGR(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrLinearLQ(uintptr_t * data, int * c);
CUDA_DEVICE bool cstrLinearLE(uintptr_t * data, int * c);

#endif
