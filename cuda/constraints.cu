#include <cuda/constraints.hh>
#include <cuda/kernels.hh>
#include <cuda/helper.hh>
#include <string.h>
#include <stdio.h>
#include <assert.h>

CUDA_DEVICE uintptr_t * cstrData;
CUDA_DEVICE uint32_t    cstrNb;

CUDA_DEVICE cstrFuncPtr     cstrTable[64] = {
        &cstrEq,       NULL,          NULL,          NULL,
        NULL,          NULL,          NULL,          NULL,

        &cstrAndNQ,    &cstrAndEQ,    &cstrAndLQ,    &cstrAndLE,
        &cstrAndGQ,    &cstrAndGR,    NULL,          NULL,

        &cstrOrNQ,     &cstrOrEQ,     &cstrOrLQ,     &cstrOrLE,
        &cstrOrGQ,     &cstrOrGR,     NULL,          NULL,

        &cstrImpNQ,    &cstrImpEQ,    &cstrImpLQ,    &cstrImpLE,
        &cstrImpGQ,    &cstrImpGR,    NULL,          NULL,

        &cstrXorNQ,    &cstrXorEQ,    &cstrXorLQ,    &cstrXorLE,
        &cstrXorGQ,    &cstrXorGR,    NULL,          NULL,

        &cstrPlusNQ,   &cstrPlusEQ,   &cstrPlusLQ,   &cstrPlusLE,
        &cstrPlusGQ,   &cstrPlusGR,   NULL,          NULL,

        &cstrTimesNQ,  &cstrTimesEQ,  &cstrTimesLQ,  &cstrTimesLE,
        &cstrTimesGQ,  &cstrTimesGR,  NULL,          NULL,

        &cstrLinearNQ, &cstrLinearEQ, &cstrLinearLQ, &cstrLinearLE,
        &cstrLinearGQ, &cstrLinearGR, NULL,          NULL
};

CUDA_DEVICE bool cstrEq(uintptr_t * data, int * c) {
    size_t v0 = (size_t) data[0];
    int    val = uint2int((unsigned int) data[1]);

    return c[v0] == val;
}

CUDA_DEVICE bool cstrAndEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrAndNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrAndGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrAndGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrAndLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrAndLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opAnd(p0, c[v0], p1, c[v1]) <= c[v2];
}


CUDA_DEVICE bool cstrOrEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrOrNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrOrGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrOrGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrOrLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrOrLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opOr(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrImpEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrImpNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrImpGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrImpGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrImpLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrImpLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opImp(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrXorEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrXorNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrXorGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrXorGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrXorLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrXorLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opXor(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrPlusEQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrPlusNQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrPlusGQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrPlusGR(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrPlusLQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrPlusLE(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return opPlus(n0, c[v0], n1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrTimesEQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrTimesNQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrTimesGQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrTimesGR(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrTimesLQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrTimesLE(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return opTimes(n, c[v0], c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrLinearEQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum == c[v0];
}

CUDA_DEVICE bool cstrLinearNQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum != c[v0];
}

CUDA_DEVICE bool cstrLinearGQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum > c[v0];
}

CUDA_DEVICE bool cstrLinearGR(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum >= c[v0];
}

CUDA_DEVICE bool cstrLinearLQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum < c[v0];
}

CUDA_DEVICE bool cstrLinearLE(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    opLinear(v, size, sum);
    return sum <= c[v0];
}
