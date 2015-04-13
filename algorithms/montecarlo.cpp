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
    mNbBinderVars = 0;
}
MonteCarlo::~MonteCarlo() { }

void MonteCarlo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType type, int min, int max) {
    if (mNbBinderVars != mNbVars) {
        mVars.resize(mNbVars+1);
        mVars[mNbVars] = mVars[mNbBinderVars];
        mVars[mNbBinderVars] = {idx, q, name, type, min, max};
    } else
        mVars.push_back({idx, q, name, type, min, max});
    mNbVars++;
    mNbBinderVars++;

    std::vector<int> varConflicts(max-min+1,0);
    mConflicts.push_back(varConflicts);
}
void MonteCarlo::newAuxVarCreated(const std::string& name, TVarType type, int min, int max) {
    if ((mLinearConstraints.size() != 0) ||
        (mTimesConstraints.size() != 0)) {
        OSTREAM << "Can't create auxiliary variable after posted constraint" << std::endl;
        GECODE_NEVER
    }
    mVars.push_back({-1, EXISTS, name, type, min, max});
    mNbVars++;
}
void MonteCarlo::newChoice(int, int, int) { }
void MonteCarlo::newPromisingScenario(const TScenario&) { }
void MonteCarlo::strategyFound() { }
void MonteCarlo::newFailure() { }
void MonteCarlo::globalFailure() { }

int MonteCarlo::getIdxVar(const std::string& name) const {
    int i = 0;
    for (const auto& v : mVars) {
        if (name == v.name) return i;
        i++;
    }
    return -1;
}

void MonteCarlo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    if (cmp != CMP_EQ) {
        OSTREAM << "Only == is implemented for times constraint" << std::endl;
        GECODE_NEVER
    }
    TConstraint newConstraint;
    newConstraint.resize(3);
    int iVar = getIdxVar(v0);
    if (iVar == -1) {
        OSTREAM << "Variable '" << v0 << "' is not defined" << std::endl;
        GECODE_NEVER
    }
    newConstraint[0] = { n, iVar };
    iVar = getIdxVar(v1);
    if (iVar == -1) {
        OSTREAM << "Variable '" << v1 << "' is not defined" << std::endl;
        GECODE_NEVER
    }
    newConstraint[1] = { 1, iVar };
    iVar = getIdxVar(v2);
    if (iVar == -1) {
        OSTREAM << "Variable '" << v2 << "' is not defined" << std::endl;
        GECODE_NEVER
    }
    newConstraint[2] = { -1, iVar };
    mLinearConstraints.push_back(newConstraint);
}

void MonteCarlo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    if (cmp != CMP_EQ) {
        OSTREAM << "Only == is implemented for linear constraint" << std::endl;
        GECODE_NEVER
    }
    TConstraint newConstraint;
    newConstraint.resize(poly.size() + 1);
    int iVar = -1;
    int i = 0;
    for (const auto& m : poly) {
        iVar = getIdxVar(m.varName);
        if (iVar == -1) {
            OSTREAM << "Variable '" << m.varName << "' is not defined" << std::endl;
            GECODE_NEVER
        }
        newConstraint[i] = { m.coeff, iVar };
        i++;
    }
    iVar = getIdxVar(v0);
    if (iVar == -1) {
        OSTREAM << "Variable '" << v0 << "' is not defined" << std::endl;
        GECODE_NEVER
    }
    newConstraint[i] = { -1, iVar };
    mLinearConstraints.push_back(newConstraint);
}

unsigned long int MonteCarlo::evalConstraints(const std::vector<int>& instance) {
    unsigned long int error = 0;
    unsigned long int v;
    // Eval times constraints
    for (const auto& constraint : mTimesConstraints) {
        v += abs(constraint[0].coeff * instance[constraint[0].iVar] \
            * instance[constraint[1].iVar] - instance[constraint[2].iVar]);
        if (v != 0) {
            mConflicts[constraint[0].iVar][instance[constraint[0].iVar]-mVars[constraint[0].iVar].dom.min]++;
            mConflicts[constraint[1].iVar][instance[constraint[1].iVar]-mVars[constraint[1].iVar].dom.min]++;
            mConflicts[constraint[2].iVar][instance[constraint[2].iVar]-mVars[constraint[2].iVar].dom.min]++;
            error += v;
        }
    }

    // Eval linear constraints
    for (const auto& constraint : mLinearConstraints) {
        v = 0;
        for (const auto& m : constraint)
            v += m.coeff * instance[m.iVar];
        if (v != 0) {
            for (const auto& m : constraint)
                mConflicts[m.iVar][instance[m.iVar]-mVars[m.iVar].dom.min]++;
            error += abs(v);
        }
    }

    return error;
}

void MonteCarlo::generateInstance(std::vector<int>& instance) {
    int i = 0;
    for (auto& v : instance) {
        v = mVars[i].dom.min + rand() % (mVars[i].dom.max - mVars[i].dom.min + 1);
        i++;
    }
}

void MonteCarlo::parallelTask() {
    OSTREAM << "MonteCarlo start" << std::endl;
    unsigned long int nbIterations = 0;
    srand(time(NULL));
    std::vector<int> instance(mNbVars);
    unsigned long int error;
    for ( ; ; ) {
        if (mainThreadFinished()) break;
        nbIterations++;
        generateInstance(instance);
        error = evalConstraints(instance);
        OSTREAM << "Error: " << error << std::endl;
        Gecode::Support::Thread::sleep(300);
    }
    OSTREAM << "MonteCarlo stop" << std::endl;
    OSTREAM << "NbIterations: " << nbIterations << std::endl;
}
