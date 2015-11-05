#ifndef __CONSTRAINTS_H_
#define __CONSTRAINTS_H_

#include <quacode/asyncalgotypes.hh>
#include <algorithms/kernels.hh>

struct Constraint {
    typedef bool (*cmpFuncPtr)(int, int);
    typedef bool (*opFuncPtr)(bool, bool, bool, bool);

    CUDA_HOST CUDA_DEVICE virtual ~Constraint() {};
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int * c) = 0;

    CUDA_HOST CUDA_DEVICE static bool cmpEQ(int, int);
    CUDA_HOST CUDA_DEVICE static bool cmpNQ(int, int);
    CUDA_HOST CUDA_DEVICE static bool cmpGQ(int, int);
    CUDA_HOST CUDA_DEVICE static bool cmpGR(int, int);
    CUDA_HOST CUDA_DEVICE static bool cmpLQ(int, int);
    CUDA_HOST CUDA_DEVICE static bool cmpLE(int, int);
    CUDA_HOST CUDA_DEVICE static cmpFuncPtr getCmpPtr(TComparisonType);
};

// v0 == v1
struct CstrEq: Constraint {
    size_t v0;
    int v2;

    CUDA_HOST CUDA_DEVICE CstrEq(size_t v0, int v2);
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int *);
};

// p0v0 op p1v1 cmp v2
struct CstrBool: Constraint {
    bool        p0;
    size_t      v0;
    opFuncPtr   op;
    bool        p1;
    size_t      v1;
    cmpFuncPtr  cmp;
    size_t      v2;

    CUDA_HOST CUDA_DEVICE CstrBool(bool p0, size_t v0, opFuncPtr op, bool p1, size_t v1, cmpFuncPtr cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int *);

    CUDA_HOST CUDA_DEVICE static bool opAnd(bool, bool, bool, bool);
    CUDA_HOST CUDA_DEVICE static bool opOr(bool, bool, bool, bool);
    CUDA_HOST CUDA_DEVICE static bool opImp(bool, bool, bool, bool);
    CUDA_HOST CUDA_DEVICE static bool opXor(bool, bool, bool, bool);
};

// n0*v0 + n1*v1 cmp v2
struct CstrPlus: Constraint {
    int n0;
    size_t v0;
    int n1;
    size_t v1;
    cmpFuncPtr cmp;
    size_t v2;

    CUDA_HOST CUDA_DEVICE CstrPlus(int n0, size_t v0, int n1, size_t v1, cmpFuncPtr cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int *);
};

// n*v0*v1 cmp v2
struct CstrTimes: Constraint {
    int n;
    size_t v0;
    size_t v1;
    cmpFuncPtr cmp;
    size_t v2;

    CUDA_HOST CUDA_DEVICE CstrTimes(int n, size_t v0, size_t v1, cmpFuncPtr cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int *);
};

struct CstrLinear: Constraint {
    const size_t * poly;
    size_t polySize;
    cmpFuncPtr cmp;
    size_t v0;

    CUDA_HOST CUDA_DEVICE ~CstrLinear();
    CUDA_HOST CUDA_DEVICE CstrLinear(const size_t * poly, size_t size, cmpFuncPtr cmp, size_t v0);
    CUDA_HOST CUDA_DEVICE virtual bool evaluate(const int *);
};

#endif
