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
#include <algorithms/montecarlo.hh>
#define OSTREAM std::cerr

MonteCarlo::MonteCarlo(bool killThread) : AsyncAlgo(killThread) {
    mNbVars = 0;
}
MonteCarlo::~MonteCarlo() { }

void MonteCarlo::newVarCreated(int, Gecode::TQuantifier, const std::string& name, TVarType, int min, int max) {
    mNbVars++;
    mVarNames.push_back(name);
    mDomains.push_back({min, max});
}
void MonteCarlo::newAuxVarCreated(const std::string& name, TVarType, int min, int max) {
    mNbVars++;
    mVarNames.push_back(name);
    mDomains.push_back({min, max});
}
void MonteCarlo::newChoice(int, int, int) { }
void MonteCarlo::newPromisingScenario(const TScenario&) { }
void MonteCarlo::strategyFound() { }
void MonteCarlo::newFailure() { }
void MonteCarlo::globalFailure() { }

int MonteCarlo::getIdxVar(const std::string& name) const {
    int i = 0;
    for (auto& vName : mVarNames) {
        if (name == vName) return i;
        i++;
    }
    return -1;
}

void MonteCarlo::postedTimes(int, std::string, std::string, TComparisonType, std::string) {
}

void MonteCarlo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, std::string oName) {
    if (cmp != CMP_EQ) {
        OSTREAM << "Only == is implemented for linear constraint" << std::endl;
        GECODE_NEVER
    }
    TConstraint newConstraint;
    newConstraint.reserve(poly.size() + 1);
    int iVar = -1;
    int i = 0;
    for (auto& m : poly) {
        iVar = getIdxVar(m.varName);
        if (iVar == -1) {
            OSTREAM << "Variable '" << m.varName << "' is not defined" << std::endl;
            GECODE_NEVER
        }
        newConstraint[i] = { m.coeff, iVar };
        i++;
    }
    iVar = getIdxVar(oName);
    if (iVar == -1) {
        OSTREAM << "Variable '" << oName << "' is not defined" << std::endl;
        GECODE_NEVER
    }
    newConstraint[i] = { -1, iVar };
    mConstraints.push_back(newConstraint);
}

int MonteCarlo::evalConstraints(const std::vector<int>& instance) const {
    return 0;
}

void MonteCarlo::parallelTask() {
    OSTREAM << "THREAD start" << std::endl;
    srand(time(NULL));
    for ( ; ; ) {
        if (mainThreadFinished()) break;
        int iVar = rand() % mNbVars;
        Gecode::Support::Thread::sleep(300);
    }
    OSTREAM << "THREAD stop" << std::endl;
}
