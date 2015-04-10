/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *    Vincent Barichard, 2015
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

#include <cassert>

forceinline bool AsyncAlgo::mainThreadFinished() const {
    return mbMainThreadFinished;
}

forceinline void AsyncAlgo::closeModeling() {
    Gecode::Support::Thread::run(this);
}

forceinline void AsyncAlgo::newVar(Gecode::TQuantifier q, std::string name, TVarType t, TVal v) {
    mBinderDesc.push_back({ .q = q, .name = name, .type = t, .dom = v });
    std::vector<int> domain;
    std::queue<Tuple> queue;
    mSwapQueues.push_back(queue);

    switch (v.type) {
        case VAL_NONE:
            GECODE_NEVER;
        case VAL_BOOL:
            domain.resize(1);
            domain[0] = v.val.b;
            break;
        case VAL_INT:
            domain.resize(1);
            domain[0] = v.val.z;
            break;
        case VAL_INTERVAL:
            domain.resize(v.val.bounds[1] - v.val.bounds[0] + 1);
            for(int i=0,x=v.val.bounds[0]; x <= v.val.bounds[1]; x++,i++)
                domain[i] = x;
            break;
    }
    mDomains.push_back(domain);
    this->newVarCreated(mDomains.size()-1,q,name,t,v);
}

forceinline void AsyncAlgo::newAuxVar(std::string name, TVarType t, TVal v) {
    mAuxVarDesc.push_back({ .q = EXISTS, .name = name, .type = t, .dom = v });
    this->newAuxVarCreated(name,t,v);
}

forceinline void AsyncAlgo::postPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2) {
    this->postedPlus(n0,v0,n1,v1,cmp,v2);
}

forceinline void AsyncAlgo::postTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2) {
    this->postedTimes(n,v0,v1,cmp,v2);
}

forceinline void AsyncAlgo::postLinear(const std::vector<Monom>& poly, TComparisonType cmp, std::string v0) {
    this->postedLinear(poly,cmp,v0);
}

forceinline void AsyncAlgo::sendSwapAsk(unsigned int iVar, unsigned int iV0, unsigned int iV1) {
    Gecode::Support::Lock lck(mSwapListsMutex);
    mSwapQueues[iVar].push({ .iV0 = iV0, .iV1 = iV1});
}

forceinline void AsyncAlgo::applySwaps(unsigned int iVar) {
    Gecode::Support::Lock lck(mSwapListsMutex);
    std::queue<Tuple> &swapQueue = mSwapQueues[iVar];
    while (!swapQueue.empty()) {
        Tuple t = swapQueue.front();
        int aux = mDomains[iVar][t.iV0];
        mDomains[iVar][t.iV0] = mDomains[iVar][t.iV1];
        mDomains[iVar][t.iV1] = aux;
        swapQueue.pop();
    }
}

