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
#ifdef QUACODE_USE_CUDA
#include <cstdlib>
#include <ctime>
#include <algorithms/gen.hh>
#include <cuda/constraints.hh>
#include <cuda/helper.hh>
#define OSTREAM std::cerr
#define MAX_VARS_IN_INTERVAL 24
#define MUTATION_THRESHOLD 1.0

// #define DEBUG

#ifdef DEBUG
#define LOG(__s) {__s}

std::string printComp(TComparisonType cmp) {
    switch (cmp) {
        case CMP_NQ:
            return "!=";
        case CMP_EQ:
            return "==";
        case CMP_LQ:
            return "<";
        case CMP_LE:
            return "<=";
        case CMP_GQ:
            return ">";
        case CMP_GR:
            return ">=";
    }
    return "NaC";
}

std::string printType(TVarType t) {
    return (t ? "Int" : "Bool");
}
#else
#define LOG(__s) /* void */
#endif

void GenAlgo::VarDescArray::push_back(const GenAlgo::VarDesc & v) {
    assert(next < GEN_MAX_VAR);

    idxInBinder[next] = v.idxInBinder;
    q[next] = v.q;
    name[next] = v.name;
    type[next] = v.type;
    dom[2 * next] = v.dom.min;
    dom[2 * next + 1] = v.dom.max;
    curDom[2 * next] = v.curDom.min;
    curDom[2 * next + 1] = v.curDom.max;

    ++next;
}

void GenAlgo::restaureDomaines(int from, int to) {
    for (int i = from + 1; i <= to; ++i) {
        mVars.curDom[2 * i] = mVars.dom[2 * i];
        mVars.curDom[2 * i + 1] = mVars.dom[2 * i + 1];
    }
}

GenAlgo::GenAlgo() : AsyncAlgo() {
    mNbVars = 0;
    mNbBinderVars = 0;
    mLastChoice = 0;
    mDomChanged = false;
}

GenAlgo::~GenAlgo() {
    mbQuacodeThreadFinished = true;

    mDestructor.acquire();
    mDestructor.release();
}

void GenAlgo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New var " << idx << "/" <<  printType(t) << " : " << name << ", type " << (q == EXISTS ? "E" : "F") << ", dom {" << min << "," << max << "}" <<std::endl;)
    mVars.push_back({idx, q, name, t, min, max, min, max});
    ++mNbVars;
    ++mNbBinderVars;
}

void GenAlgo::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New auxiliary var " << name << "/" << printType(t) << ", dom {" << min << "," << max << "}" << std::endl;)
    mVars.push_back({-1, EXISTS, name, t, min, max, min, max});
    ++mNbVars;
}

void GenAlgo::newChoice(int iVar, int min, int max) {
    LOG(OSTREAM << "Chef ! We need to explore others choices : " << iVar << " {" << min << "," << max << "}" << std::endl;)

    mDomaine.acquire();

    if (iVar < mLastChoice) {
        restaureDomaines(iVar, mLastChoice);
    }

    mVars.curDom[iVar * 2] = min;
    mVars.curDom[iVar * 2 + 1] = max;
    mLastChoice = iVar;
    mDomChanged = true;

    mDomaine.release();
}

void GenAlgo::newPromisingScenario(const TScenario& scenario) {
    LOG(
    OSTREAM << "Chef ! I think this scenario is interesting : ";
    for (auto s: scenario) {
        OSTREAM << "{" << s.min << "," << s.max << "}, ";
    }
    OSTREAM << std::endl;
    )

    if (scenario.size()) return;
}

void GenAlgo::strategyFound() {
    LOG(OSTREAM << "Chef ! We just found a solution !" << std::endl;)
}

void GenAlgo::newFailure() {
    LOG(OSTREAM << "Another faillure chef !" << std::endl;)
}

void GenAlgo::globalFailure() {
    LOG(OSTREAM << "It can't be done. ABORT ! ABORT ! ABORT !" << std::endl;)
}

/// Function called when a new 'v0 == v2' constraint is posted
void GenAlgo::postedEq(const std::string& v0, int val) {
    LOG(OSTREAM << "New constraint Eq " << v0 << "=" << val << std::endl;)
    size_t v0Idx = findVar(v0);

    if (v0Idx != (size_t)-1) {
		mCstrs.insert(mCstrs.end(), {
				CSTR_EQ_IDX, v0Idx, int2uint(val), NULL,
				NULL, NULL, NULL, NULL
				});
    }
    else {
        OSTREAM << "Can't find " << v0 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint And " << p0 << ":" << v0 << " && " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_AND_IDX | cmp, p0, v0Idx, p1,
                v1Idx, v2Idx, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Or " << p0 << ":" << v0 << " || " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_OR_IDX | cmp, p0, v0Idx, p1,
                v1Idx, v2Idx, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Imp " << p0 << ":" << v0 << " >> " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_IMP_IDX | cmp, p0, v0Idx, p1,
                v1Idx, v2Idx, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Xor " << p0 << ":" << v0 << " ^ " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_XOR_IDX | cmp, p0, v0Idx, p1,
                v1Idx, v2Idx, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
void GenAlgo::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Plus " << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_PLUS_IDX | cmp, int2uint(n0), v0Idx, int2uint(n1),
                v1Idx, v2Idx, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
void GenAlgo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Times " << n << "*" << v0 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrs.insert(mCstrs.end(), {
                CSTR_TIMES_IDX | cmp, int2uint(n), v0Idx, v1Idx,
                v2Idx, NULL, NULL, NULL
                });
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
void GenAlgo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    LOG(
            OSTREAM << "New constraint Linear ";
            for (auto m : poly) {
            OSTREAM << m.coeff << "*" << m.varName << " + ";
            }
            OSTREAM << printComp(cmp) << " " << v0 << std::endl;
       )

        size_t v0Idx = findVar(v0);

    if (v0Idx != (size_t)-1) {
        size_t * h_polyCpy = new size_t[poly.size() * 2], * h_polyCpyStart = h_polyCpy;
        size_t d_polyCpy; // On the GPU

        for (auto it = poly.begin(); it != poly.end(); ++it) {
            size_t viIdx = findVar(it->varName);

            if (viIdx != (size_t)-1) {
                *h_polyCpy++ = it->coeff;
                *h_polyCpy++ = viIdx;
            }
            else {
                OSTREAM << "Can't find " << it->varName << std::endl;
                GECODE_NEVER
            }
        }
        d_polyCpy = pushPolyToGPU(h_polyCpyStart, poly.size() * 2);

        mCstrs.insert(mCstrs.end(), {
                        CSTR_LINEAR_IDX | cmp, d_polyCpy, poly.size(), v0Idx,
                        NULL, NULL, NULL, NULL
                        });
        delete[] h_polyCpyStart;
    }
    else {
        OSTREAM << "Can't find variable " << v0 << std::endl;
        GECODE_NEVER
    }
}


void GenAlgo::parallelTask() {
    std::vector<TVarType>               tmpTypes;
    std::vector<Gecode::TQuantifier>    tmpQuan;

    LOG(OSTREAM << "THREAD start" << std::endl;)

    OSTREAM << "Post constraints and variables to GPU" << std::endl;
    pushCstrToGPU(mCstrs.data(), mCstrs.size());
    pushVarToGPU(mVars.type, mVars.q, mVars.next);
    pushDomToGPU(mVars.curDom, mVars.next * 2);


    mDestructor.acquire();
    srand(time(NULL));

    int cpt = 0;

    while (!mbQuacodeThreadFinished) {
        size_t *results = nullptr;
        size_t resultsSize = 0;

        if (mDomChanged) {
            mDomaine.acquire();
            pushDomToGPU(mVars.curDom, mVars.next * 2);
            memcpy(mVars.savedDom, mVars.curDom, mVars.next * 2);
            mDomaine.release();
        }

        initPopulation(8192);
        doTheMagic(1000);
        getResults(&results, &resultsSize);

        ++cpt;

        // for (size_t i = 0; i < resultsSize; ++i) {
        //     OSTREAM << results[i] << ", ";
        // }
        // OSTREAM << std::endl;

        delete[] results;
    }

    OSTREAM << cpt << std::endl;

    mDestructor.release();
    LOG(OSTREAM << "THREAD stop" << std::endl;)
}


size_t GenAlgo::findVar(const std::string & name) {
    for (size_t i = 0; i < mVars.next; ++i) {
        if (mVars.name[i] == name) {
            return i;
        }
    }

    return -1;
}

#endif
