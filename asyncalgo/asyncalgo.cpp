/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *    Vincent Barichard, 2013
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include <asyncalgo/asyncalgo.hh>
#define OSTREAM std::cerr

AsyncAlgo::AsyncAlgo(bool killThread)
    : mbMainThreadExited(false), mbKillThread(killThread) {
}

AsyncAlgo::~AsyncAlgo() {
    mbMainThreadExited = true;
    if (!mbKillThread) {
        mExit.acquire();
        mExit.release();
    }
#ifdef LOG_AUDIT
    OSTREAM << "END" << std::endl;
#endif
}

void AsyncAlgo::closeModeling() {
#ifdef LOG_AUDIT
    OSTREAM << "START THREAD" << std::endl;
#endif
    Gecode::Support::Thread::run(this);
}

void AsyncAlgo::newVar(Gecode::TQuantifier q, std::string name, TVarType t, TVal v) {
    mBinderDesc.push_back({ .q = q, .name = name, .type = t, .dom = v });
    std::vector<int> domain;
    std::vector<Tuple> list;
    mSwapLists.push_back(list);

    OSTREAM << "VAR_BINDER       =";
    OSTREAM << " var(" << ((q==EXISTS)?"E":"F") << "," << ((t==TYPE_BOOL)?"B":"I") << "," << name;
    switch (v.type) {
        case VAL_NONE:
            GECODE_NEVER;
        case VAL_BOOL:
            OSTREAM << ",bool(" << v.val.b << ")";
            domain.resize(1);
            domain[0] = v.val.b;
            break;
        case VAL_INT:
            OSTREAM << ",int(" << v.val.z << ")";
            domain.resize(1);
            domain[0] = v.val.z;
            break;
        case VAL_INTERVAL:
            OSTREAM << ",interval(" << v.val.bounds[0] << ":" << v.val.bounds[1] << ")";
            domain.resize(v.val.bounds[1] - v.val.bounds[0] + 1);
            for(int i=0,x=v.val.bounds[0]; x <= v.val.bounds[1]; x++,i++)
                domain[i] = x;
            break;
    }
    OSTREAM << ")" << std::endl;
    mDomains.push_back(domain);
}

// Add a new auxiliary variable \a var
void AsyncAlgo::newAuxVar(std::string name, TVarType t, TVal v) {
    mAuxVarDesc.push_back({ .q = EXISTS, .name = name, .type = t, .dom = v });
    OSTREAM << "VAR_AUX          =";
    OSTREAM << " var(E," << ((t==TYPE_BOOL)?"B":"I") << "," << name;
    switch (v.type) {
        case VAL_NONE:
            break;
        case VAL_BOOL:
            OSTREAM << ",bool(" << v.val.b << ")";
            break;
        case VAL_INT:
            OSTREAM << ",int(" << v.val.z << ")";
            break;
        case VAL_INTERVAL:
            OSTREAM << ",interval(" << v.val.bounds[0] << ":" << v.val.bounds[1] << ")";
            break;
    }
    OSTREAM << ")" << std::endl;
}

void AsyncAlgo::newChoice(int iVar, TVal val) {
    OSTREAM << "CHOICE           = ";
    if (val.type == VAL_INTERVAL)
        OSTREAM << mBinderDesc[iVar].name << " # [" << val.val.bounds[0] << ";" << val.val.bounds[1] << "]" << std::endl;
    else if (val.type == VAL_BOOL)
        OSTREAM << mBinderDesc[iVar].name << " # " << val.val.b << std::endl;
    else
        OSTREAM << mBinderDesc[iVar].name << " # " << val.val.z << std::endl;
}
void AsyncAlgo::newPromisingScenario(const TScenario& scenario) {
    bool bFirst = true;
    OSTREAM << "PR SCENARIO      = ";
    for(auto &v : scenario) {
        if (!bFirst) OSTREAM << ", ";
        if (v.type == VAL_NONE)
            OSTREAM << "NOVALUE)";
        else if (v.type == VAL_INTERVAL)
            OSTREAM << "[" << v.val.bounds[0] << ";" << v.val.bounds[1] << "]";
        else if (v.type == VAL_BOOL)
            OSTREAM << v.val.b;
        else
            OSTREAM << v.val.z;
        bFirst = false;
    }
    OSTREAM << std::endl;
}
void AsyncAlgo::strategyFound() {
    OSTREAM << "STRATEGY FOUND" << std::endl;
}
void AsyncAlgo::newFailure() {
    OSTREAM << "FAIL" << std::endl;
}
void AsyncAlgo::globalFailure() {
    OSTREAM << "GLOBAL FAILURE" << std::endl;
}

void AsyncAlgo::postPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    OSTREAM << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void AsyncAlgo::postTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    if (n != 1) OSTREAM << n << " * ";
    OSTREAM << v0 << " * " << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void AsyncAlgo::postLinear(std::vector<Monom> poly, TComparisonType cmp, std::string v0) {
    bool bFirst = true;
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    for(auto &m : poly) {
        if (!bFirst) OSTREAM << " + ";
        OSTREAM << m.c << "*" << m.var;
        bFirst = false;
    }
    OSTREAM << " " << s_ComparisonType[cmp] << " " << v0 << std::endl;
}

void AsyncAlgo::sendSwapAsk(unsigned int iVar, unsigned int iV0, unsigned int iV1) {
    mSwapLists[iVar].push_back({ .iV0 = iV0, .iV1 = iV1});
}

void AsyncAlgo::applySwaps(unsigned int iVar) {
}

void AsyncAlgo::run() {
    if (!mbKillThread)
        mExit.acquire();
    parallelTask();
    if (!mbKillThread)
        mExit.release();
    Gecode::Support::Event dummy;
    dummy.wait();
}

void AsyncAlgo::parallelTask() {
    for ( ; ; ) {
        if (mbMainThreadExited) break;
        OSTREAM << "THREAD ..." << std::endl;
        Gecode::Support::Thread::sleep(300);
    }
}
