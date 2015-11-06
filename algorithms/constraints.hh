#ifndef __CONSTRAINTS_H_
#define __CONSTRAINTS_H_

#include <quacode/asyncalgotypes.hh>
#include <algorithms/cuda.hh>
#include <cstddef>

typedef     int     TOperatorType;
#define     OP_AND 0
#define     OP_OR  1
#define     OP_IMP 2
#define     OP_XOR 3

struct Constraint {
    typedef bool (*cmpFuncPtr)(int, int);
    typedef bool (*opFuncPtr)(bool, bool, bool, bool);

    CUDA_HOST   virtual ~Constraint() {};
    CUDA_DEVICE virtual bool evaluate(const int * c) = 0;

    CUDA_DEVICE static bool cmpEQ(int, int);
    CUDA_DEVICE static bool cmpNQ(int, int);
    CUDA_DEVICE static bool cmpGQ(int, int);
    CUDA_DEVICE static bool cmpGR(int, int);
    CUDA_DEVICE static bool cmpLQ(int, int);
    CUDA_DEVICE static bool cmpLE(int, int);
    CUDA_DEVICE static cmpFuncPtr getCmpPtr(TComparisonType);
    CUDA_HOST   static bool evaluate(Constraint **, size_t, const int *);
    CUDA_DEVICE virtual void describe();


};

// v0 == v1
struct CstrEq: Constraint {
    size_t v0;
    int v2;

    CUDA_DEVICE virtual bool evaluate(const int *);
    CUDA_DEVICE virtual void describe();

    CUDA_HOST  static CstrEq * create(size_t v0, int v2);
    CUDA_HOST CUDA_DEVICE CstrEq(size_t v0, int v2);
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

    CUDA_DEVICE static  opFuncPtr getOpPtr(TOperatorType op);

    CUDA_DEVICE virtual bool evaluate(const int *);
    CUDA_DEVICE virtual void describe();

    CUDA_DEVICE static bool opAnd(bool, bool, bool, bool);
    CUDA_DEVICE static bool opOr(bool, bool, bool, bool);
    CUDA_DEVICE static bool opImp(bool, bool, bool, bool);
    CUDA_DEVICE static bool opXor(bool, bool, bool, bool);

    CUDA_HOST   static CstrBool * create(bool p0, size_t v0, TOperatorType op, bool p1, size_t v1, TComparisonType cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE CstrBool(bool p0, size_t v0, opFuncPtr op, bool p1, size_t v1, cmpFuncPtr cmp, size_t v2);
};

// n0*v0 + n1*v1 cmp v2
struct CstrPlus: Constraint {
    int n0;
    size_t v0;
    int n1;
    size_t v1;
    cmpFuncPtr cmp;
    size_t v2;

    CUDA_DEVICE virtual bool evaluate(const int *);
    CUDA_DEVICE virtual void describe();
    CUDA_HOST   static CstrPlus* create(int n0, size_t v0, int n1, size_t v1, TComparisonType cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE CstrPlus(int n0, size_t v0, int n1, size_t v1, cmpFuncPtr cmp, size_t v2);
};

// n*v0*v1 cmp v2
struct CstrTimes: Constraint {
    int n;
    size_t v0;
    size_t v1;
    cmpFuncPtr cmp;
    size_t v2;

    CUDA_DEVICE virtual bool evaluate(const int *);
    CUDA_DEVICE virtual void describe();
    CUDA_HOST   static CstrTimes * create(int n, size_t v0, size_t v1, TComparisonType cmp, size_t v2);
    CUDA_HOST CUDA_DEVICE CstrTimes(int n, size_t v0, size_t v1, cmpFuncPtr cmp, size_t v2);
};

struct CstrLinear: Constraint {
    size_t * poly;
    size_t polySize;
    cmpFuncPtr cmp;
    size_t v0;

    CUDA_HOST ~CstrLinear();
    CUDA_DEVICE virtual bool evaluate(const int *);
    CUDA_DEVICE virtual void describe();
    CUDA_HOST   static CstrLinear * create(size_t * poly, size_t size, TComparisonType cmp, size_t v0);
    CUDA_HOST CUDA_DEVICE CstrLinear(size_t * poly, size_t size, cmpFuncPtr cmp, size_t v0);
};

#endif
