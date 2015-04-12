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

#include <cstdlib>
#include <ctime>
#include <algorithms/dumb.hh>
#define OSTREAM std::cerr

DumbAlgorithm::DumbAlgorithm(bool killThread) : AsyncAlgo(killThread) {
    mNbVars = 0;
}
DumbAlgorithm::~DumbAlgorithm() { }

void DumbAlgorithm::newVarCreated(int, Gecode::TQuantifier, const std::string& name, TVarType, int min, int max) {
    mNbVars++;
    mVarNames.push_back(name);
    mDomains.push_back({min, max});
}

void DumbAlgorithm::newAuxVarCreated(const std::string&, TVarType, int, int) { }
void DumbAlgorithm::newChoice(int, int, int) { }
void DumbAlgorithm::newPromisingScenario(const TScenario&) { }
void DumbAlgorithm::strategyFound() { }
void DumbAlgorithm::newFailure() { }
void DumbAlgorithm::globalFailure() { }

void DumbAlgorithm::postedTimes(int, const std::string&, const std::string&, TComparisonType, const std::string&) { }
void DumbAlgorithm::postedLinear(const std::vector<Monom>&, TComparisonType, const std::string&) { }

void DumbAlgorithm::parallelTask() {
    OSTREAM << "THREAD start" << std::endl;
    srand(time(NULL));
    for ( ; ; ) {
        if (mainThreadFinished()) break;
        int iVar = rand() % mNbVars;
        int i0 = rand() % (mDomains[iVar].max - mDomains[iVar].min + 1);
        int i1 = rand() % (mDomains[iVar].max - mDomains[iVar].min + 1);
        swap(iVar,i0,i1);
        OSTREAM << "Swap(" << mVarNames[iVar] << "," << i0 << "," << i1 << ")" << std::endl;
        Gecode::Support::Thread::sleep(300);
    }
    OSTREAM << "THREAD stop" << std::endl;
}
