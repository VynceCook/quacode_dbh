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
#include <algorithms/test.hh>
#define OSTREAM std::cerr

//#define DEBUG

#ifdef DEBUG
#define LOG(__s) __s
#else
#define LOG(__s) /* void */
#endif

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

void TestAlgo::restaureDomaines(int from, int to) {
    for (int i = from + 1; i <= to; ++i) {
        mVars[i].curDom = mVars[i].dom;
    }
}

TestAlgo::TestAlgo() : AsyncAlgo() {
    mNbVars = 0;
    mNbBinderVars = 0;
    mLastChoice = 0;
}

TestAlgo::~TestAlgo() {
    mbQuacodeThreadFinished = true;

    mDestructor.acquire();
    mDestructor.release();
}

void TestAlgo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    LOG({OSTREAM << "New var " << idx << "/" <<  printType(t) << " : " << name << ", type " << (q == EXISTS ? "E" : "F") << ", dom {" << min << "," << max << "}" <<std::endl});
    mVars.push_back({idx, q, name, t, min, max, min, max});
    ++mNbVars;
    ++mNbBinderVars;
}

void TestAlgo::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    LOG({OSTREAM << "New auxiliary var " << name << "/" << printType(t) << ", dom {" << min << "," << max << "}" << std::endl});
    mVars.push_back({-1, EXISTS, name, t, min, max, min, max});
    ++mNbVars;
}

void TestAlgo::newChoice(int iVar, int min, int max) {
    LOG({OSTREAM << "Chef ! We need to explore others choices : " << iVar << " {" << min << "," << max << "}" << std::endl});

    if (iVar < mLastChoice) {
        restaureDomaines(iVar, mLastChoice);
    }

    mVars[iVar].curDom = {min, max};
    mLastChoice = iVar;

    LOG({
    for(auto s : mVars) {
        OSTREAM << "{" << s.curDom.min << "," << s.curDom.max << "}" << ", ";
    }
    OSTREAM << std::endl;})
}

void TestAlgo::newPromisingScenario(const TScenario& scenario) {
    LOG({
    OSTREAM << "Chef ! I think this scenario is interesting : ";
    for (auto s: scenario) {
        OSTREAM << "{" << s.min << "," << s.max << "}, ";
    }
    OSTREAM << std::endl;
    })
}

void TestAlgo::strategyFound() {
    LOG({OSTREAM << "Chef ! We just found a solution !" << std::endl});
}

void TestAlgo::newFailure() {
    LOG({OSTREAM << "Another faillure chef !" << std::endl});
}

void TestAlgo::globalFailure() {
    LOG({OSTREAM << "It can't be done. ABORT ! ABORT ! ABORT !" << std::endl});
}

/// Function called when a new 'v0 == v2' constraint is posted
void TestAlgo::postedEq(const std::string& v0, int val) {
    LOG({OSTREAM << "New constraint Eq " << v0 << "=" << val << std::endl});
    VarDesc* v0ptr = findVar(v0);

    if (v0ptr) {
        mCstrEq.push_back({v0ptr, val});
    }
    else {
        OSTREAM << "Can't find " << v0 << ", we ignore this constraint." << std::endl;
    }
}

/// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint And " << p0 << ":" << v0 << " && " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrBool.push_back({p0, v0ptr, OP_AND, p1, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}

/// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint Or " << p0 << ":" << v0 << " || " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrBool.push_back({p0, v0ptr, OP_OR, p1, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}

/// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint Imp " << p0 << ":" << v0 << " >> " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrBool.push_back({p0, v0ptr, OP_IMP, p1, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}

/// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint Xor " << p0 << ":" << v0 << " ^ " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrBool.push_back({p0, v0ptr, OP_XOR, p1, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}


/// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
void TestAlgo::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint Plus " << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrPlus.push_back({n0, v0ptr, n1, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}

/// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
void TestAlgo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG({OSTREAM << "New constraint Times " << n << "*" << v0 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl});
    VarDesc * v0ptr = findVar(v0), *v1ptr = findVar(v1), *v2ptr = findVar(v2);

    if (v0ptr && v1ptr && v2ptr) {
        mCstrTimes.push_back({n, v0ptr, v1ptr, cmp, v2ptr});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
    }
}

/// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
void TestAlgo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    LOG({
    OSTREAM << "New constraint Linear ";
    for (auto m : poly) {
        OSTREAM << m.coeff << "*" << m.varName << " + ";
    }
    OSTREAM << printComp(cmp) << " " << v0 << std::endl;
    })

    VarDesc * v0ptr = findVar(v0);

    if (v0ptr) {
        mCstrLinear.push_back({{}, cmp, v0ptr});

        for (auto it = poly.begin(); it != poly.end(); ++it) {
            VarDesc * viptr = findVar(it->varName);

            if (viptr) {
                mCstrLinear.back().poly.push_back(std::make_pair(it->coeff, findVar(it->varName)));
            }
            else {
                OSTREAM << "Can't find " << it->varName << std::endl;
                mCstrLinear.pop_back();
                return;
            }
        }
    }
    else {
        OSTREAM << "Can't find variable " << v0 << std::endl;
    }
}


void TestAlgo::parallelTask() {
    LOG({OSTREAM << "THREAD start" << std::endl});

    mDestructor.acquire();
    srand(time(NULL));
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;

        Gecode::Support::Thread::sleep(300);
    }

    mDestructor.release();
    LOG({OSTREAM << "THREAD stop" << std::endl});
}


TestAlgo::VarDesc* TestAlgo::findVar(const std::string & name) {
    for (auto it = mVars.begin(); it != mVars.end(); ++it)
        if (it->name == name) return &(*it);

    return nullptr;
}
