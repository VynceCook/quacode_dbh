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

MonteCarlo::MonteCarlo() : AsyncAlgo(), mbQuacodeThreadFinished(false) {
    mNbVars = 0;
    mNbBinderVars = 0;
}
MonteCarlo::~MonteCarlo() {
    mbQuacodeThreadFinished = true;
    // We block the destructor until the background thread finished
    mDestructor.acquire();
    mDestructor.release();
}

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
            if (constraint[0].iVar < mNbBinderVars)
                mConflicts[constraint[0].iVar][instance[constraint[0].iVar]-mVars[constraint[0].iVar].dom.min]++;
            if (constraint[1].iVar < mNbBinderVars)
                mConflicts[constraint[1].iVar][instance[constraint[1].iVar]-mVars[constraint[1].iVar].dom.min]++;
            if (constraint[2].iVar < mNbBinderVars)
                mConflicts[constraint[2].iVar][instance[constraint[2].iVar]-mVars[constraint[2].iVar].dom.min]++;
            error += 3;
            //error += v;
        }
    }

    // Eval linear constraints
    for (const auto& constraint : mLinearConstraints) {
        v = 0;
        for (const auto& m : constraint)
            v += m.coeff * instance[m.iVar];
        if (v != 0) {
            for (const auto& m : constraint)
                if (m.iVar < mNbBinderVars) {
                    mConflicts[m.iVar][instance[m.iVar]-mVars[m.iVar].dom.min]++;
                    error++;
                }
            //error += abs(v);
        }
    }

    return error;
}

int MonteCarlo::getIdxMinConflicts() {
    assert(mConflicts.size() > 0);

    int error = 0;
    int idx = 0;
    int minError = 0;
    for (auto& v : mConflicts[0])
        minError += v;

    for (unsigned int i=0; i < mConflicts.size(); i++) {
        error = 0;
        for (auto& v : mConflicts[i])
            error += v;
        if (error < minError) {
            minError = error;
            idx = i;
        }
    }
    return idx;
}

void MonteCarlo::parallelTask() {
    // If mainThread has finished, member variables are not usable anymore
    // so we block the destructor until we finish here
    mDestructor.acquire();

    OSTREAM << "MonteCarlo start" << std::endl;
    srand(time(NULL));
    unsigned long int nbIterations = 0;
    std::vector<int> instance(mNbVars);
    unsigned long int error, nError;
    unsigned long int countFreq = 0, freq = 1000;
    int k, vSaved;

    countFreq = 0;
    // Generate first random instance
    k = 0;
    for (auto& v : instance) {
        v = mVars[k].dom.min + rand() % (mVars[k].dom.max - mVars[k].dom.min + 1);
        k++;
    }
    error = evalConstraints(instance);
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;
        // Select next variable
        //k = getIdxMinConflicts();
        k = rand() % mNbBinderVars;
        // Randomly choose a new value
        vSaved = instance[k];
        instance[k] = mVars[k].dom.min + rand() % (mVars[k].dom.max - mVars[k].dom.min + 1);
        nbIterations++;
        countFreq++;
        nError = evalConstraints(instance);
        if (nError > error) {
            error = nError;
            OSTREAM << "New error: " << nError << std::endl;
        } else {
            error = nError;
            instance[k] = vSaved;
        }
        if (countFreq == freq) {
            OSTREAM << "Frequency threshold " << freq << " reached" << std::endl;
            countFreq = 0;
        }
    }

    for (auto& varConflicts : mConflicts) {
        for (auto& v : varConflicts)
            OSTREAM << v << " ";
        OSTREAM << std::endl;
    }
    OSTREAM << "MonteCarlo stop" << std::endl;
    OSTREAM << "NbIterations: " << nbIterations << std::endl;
    // Release the destructor as we have finish everything
    mDestructor.release();
}
