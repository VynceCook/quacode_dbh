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

#include <algorithms/dumb.hh>
#define OSTREAM std::cerr

DumbAlgorithm::DumbAlgorithm(bool killThread) : AsyncAlgo(killThread) {
    mNbVars = 0;
}
DumbAlgorithm::~DumbAlgorithm() { }

void DumbAlgorithm::newVarCreated(int, Gecode::TQuantifier, std::string, TVarType, int, int) {
    mNbVars++;
}

void DumbAlgorithm::newAuxVarCreated(std::string, TVarType, int, int) { }
void DumbAlgorithm::newChoice(int, int, int) { }
void DumbAlgorithm::newPromisingScenario(const TScenario&) { }
void DumbAlgorithm::strategyFound() { }
void DumbAlgorithm::newFailure() { }
void DumbAlgorithm::globalFailure() { }

void DumbAlgorithm::postedPlus(int, std::string, int, std::string, TComparisonType, std::string) { }
void DumbAlgorithm::postedTimes(int, std::string, std::string, TComparisonType, std::string) { }
void DumbAlgorithm::postedLinear(const std::vector<Monom>&, TComparisonType, std::string) { }

void DumbAlgorithm::parallelTask() {
    OSTREAM << "THREAD start" << std::endl;
    for ( ; ; ) {
        if (mainThreadFinished()) break;
        OSTREAM << "THREAD ..." << std::endl;
//        swap(2,8,26);
        Gecode::Support::Thread::sleep(300);
        break;
    }
    OSTREAM << "THREAD stop" << std::endl;
}
