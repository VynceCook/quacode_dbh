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

forceinline void AsyncAlgo::closeModeling() {
    Gecode::Support::Thread::run(this); // Start the run member function of AsyncAlgo
}

forceinline void AsyncAlgo::newVar(Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    std::vector<int> domain;
    mDomainsMutex.push_back(new Gecode::Support::Mutex());

    domain.resize(max - min + 1);
    for(int i=0,x=min; x <= max; x++,i++)
        domain[i] = x;

    mDomains.push_back(domain);
    this->newVarCreated(mDomains.size()-1,q,name,t,min,max);
}

forceinline void AsyncAlgo::newAuxVar(const std::string& name, TVarType t, int min, int max) {
    this->newAuxVarCreated(name,t,min,max);
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

forceinline void AsyncAlgo::swap(unsigned int iVar, unsigned int iV0, unsigned int iV1) {
    Gecode::Support::Lock lck(*mDomainsMutex[iVar]);
    int aux = mDomains[iVar][iV0];
    mDomains[iVar][iV0] = mDomains[iVar][iV1];
    mDomains[iVar][iV1] = aux;
}

forceinline void AsyncAlgo::copyDomainIf(int iVar, const Gecode::Int::IntView& iv, std::vector<int>& dest) const {
    Gecode::Support::Lock lck(*mDomainsMutex[iVar]);
    for (const auto& x : mDomains[iVar])
        if (iv.in(x))
            dest.push_back(x);
}

forceinline void AsyncAlgo::copyDomain(int iVar, std::vector<int>& dest) const {
    Gecode::Support::Lock lck(*mDomainsMutex[iVar]);
    dest.assign(mDomains[iVar].begin(),mDomains[iVar].end());
}

