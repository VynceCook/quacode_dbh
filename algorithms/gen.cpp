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
#include <algorithms/gen.hh>
#include <algorithms/kernels.hh>
#define OSTREAM std::cerr

//#define DEBUG

#ifdef DEBUG
#define LOG(__s) {__s}
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

void GenAlgo::restaureDomaines(int from, int to) {
    for (int i = from + 1; i <= to; ++i) {
        mVars[i].curDom = mVars[i].dom;
    }
}

GenAlgo::GenAlgo() : AsyncAlgo() {
    mNbVars = 0;
    mNbBinderVars = 0;
    mLastChoice = 0;
}

GenAlgo::~GenAlgo() {
    mbQuacodeThreadFinished = true;

    mDestructor.acquire();
    mDestructor.release();
}

void GenAlgo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New var " << idx << "/" <<  printType(t) << " : " << name << ", type " << (q == EXISTS ? "E" : "F") << ", dom {" << min << "," << max << "}" <<std::endl);
    mVars.push_back({idx, q, name, t, min, max, min, max});
    ++mNbVars;
    ++mNbBinderVars;
}

void GenAlgo::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New auxiliary var " << name << "/" << printType(t) << ", dom {" << min << "," << max << "}" << std::endl);
    mVars.push_back({-1, EXISTS, name, t, min, max, min, max});
    ++mNbVars;
}

void GenAlgo::newChoice(int iVar, int min, int max) {
    LOG(OSTREAM << "Chef ! We need to explore others choices : " << iVar << " {" << min << "," << max << "}" << std::endl);

    if (iVar < mLastChoice) {
        restaureDomaines(iVar, mLastChoice);
    }

    mVars[iVar].curDom = {min, max};
    mLastChoice = iVar;

    LOG(
    for(auto s : mVars) {
        OSTREAM << "{" << s.curDom.min << "," << s.curDom.max << "}" << ", ";
    }
    OSTREAM << std::endl;)
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
    LOG(OSTREAM << "Chef ! We just found a solution !" << std::endl);
}

void GenAlgo::newFailure() {
    LOG(OSTREAM << "Another faillure chef !" << std::endl);
}

void GenAlgo::globalFailure() {
    LOG(OSTREAM << "It can't be done. ABORT ! ABORT ! ABORT !" << std::endl);
}

/// Function called when a new 'v0 == v2' constraint is posted
void GenAlgo::postedEq(const std::string& v0, int val) {
    LOG(OSTREAM << "New constraint Eq " << v0 << "=" << val << std::endl);
    size_t v0Idx = findVar(v0);

    if (v0Idx != (size_t)-1) {
        mCstrEq.push_back({v0Idx, val});
    }
    else {
        OSTREAM << "Can't find " << v0 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint And " << p0 << ":" << v0 << " && " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrBool.push_back({p0, v0Idx, OP_AND, p1, v1Idx, cmp, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Or " << p0 << ":" << v0 << " || " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrBool.push_back({p0, v0Idx, OP_OR, p1, v1Idx, cmp, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Imp " << p0 << ":" << v0 << " >> " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrBool.push_back({p0, v0Idx, OP_IMP, p1, v1Idx, cmp, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Xor " << p0 << ":" << v0 << " ^ " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrBool.push_back({p0, v0Idx, OP_XOR, p1, v1Idx, cmp, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}


/// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
void GenAlgo::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Plus " << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrPlus.push_back({n0, v0Idx, n1, v1Idx, cmp, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
void GenAlgo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Times " << n << "*" << v0 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl);
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        mCstrTimes.push_back({n, v0Idx, v1Idx, cmp, v2Idx});
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
        mCstrLinear.push_back({{}, cmp, v0Idx});

        for (auto it = poly.begin(); it != poly.end(); ++it) {
            size_t viIdx = findVar(it->varName);

            if (viIdx != (size_t)-1) {
                mCstrLinear.back().poly.push_back(std::make_pair(it->coeff, viIdx));
            }
            else {
                OSTREAM << "Can't find " << it->varName << std::endl;
                mCstrLinear.pop_back();
                GECODE_NEVER
            }
        }
    }
    else {
        OSTREAM << "Can't find variable " << v0 << std::endl;
        GECODE_NEVER
    }
}


void GenAlgo::parallelTask() {
    LOG(OSTREAM << "THREAD start" << std::endl);

    OSTREAM << "Calling foo kernel" << std::endl;
    foo();
    
    mDestructor.acquire();
    srand(time(NULL));
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;

        Gecode::Support::Thread::sleep(300);
    }


    mDestructor.release();
    LOG(OSTREAM << "THREAD stop" << std::endl);
}


size_t GenAlgo::findVar(const std::string & name) {
    for (size_t i = 0; i < mVars.size(); ++i)
        if (mVars[i].name == name) return i;

    return -1;
}

bool GenAlgo::evaluate(const std::vector<int> &vars) {
    for (auto it = mCstrEq.begin(); it != mCstrEq.end(); ++it) {
        if (vars[it->v0] != it->v2) return  false;
    }

    for (auto it = mCstrBool.begin(); it != mCstrBool.end(); ++it) {
        bool v0 = it->p0 ? it->v0 : !it->v0;
        bool v1 = it->p1 ? it->v1 : !it->v1;
        bool lhs, rhs = it->v2;
        bool res;

        switch (it->op) {
            case OP_AND:
                lhs = v0 && v1;
                break;

            case OP_OR:
                lhs = v0 || v1;
                break;

            case OP_IMP:
                lhs = !(v0 && !v1);
                break;

            case OP_XOR:
                lhs = (!v0 != !v1);
                break;

            default:
                GECODE_NEVER;
        }

        switch (it->cmp) {
            case CMP_NQ:
                res = (lhs != rhs);
                break;

            case CMP_EQ:
                res = (lhs == rhs);
                break;

            case CMP_LQ:
                res = (lhs < rhs);
                break;

            case CMP_LE:
                res = (lhs <= rhs);
                break;

            case CMP_GQ:
                res = (lhs > rhs);
                break;

            case CMP_GR:
                res = (lhs >= rhs);
                break;

            default:
                GECODE_NEVER
        }

        if (!res) return false;
    }

    for (auto it = mCstrPlus.begin(); it != mCstrPlus.end(); ++it) {
        int lhs = it->n0 * vars[it->v0] + it->n1 * vars[it->v1], rhs = vars[it->v2];
        bool res;

        switch (it->cmp) {
            case CMP_NQ:
                res = (lhs != rhs);
                break;

            case CMP_EQ:
                res = (lhs == rhs);
                break;

            case CMP_LQ:
                res = (lhs < rhs);
                break;

            case CMP_LE:
                res = (lhs <= rhs);
                break;

            case CMP_GQ:
                res = (lhs > rhs);
                break;

            case CMP_GR:
                res = (lhs >= rhs);
                break;

            default:
                GECODE_NEVER
        }
        if (!res) return false;
    }

    for (auto it = mCstrTimes.begin(); it != mCstrTimes.end(); ++it) {
        int lhs = it->n * vars[it->v0] * vars[it->v1], rhs = vars[it->v2];
        bool res;

        switch (it->cmp) {
            case CMP_NQ:
                res = (lhs != rhs);
                break;

            case CMP_EQ:
                res = (lhs == rhs);
                break;

            case CMP_LQ:
                res = (lhs < rhs);
                break;

            case CMP_LE:
                res = (lhs <= rhs);
                break;

            case CMP_GQ:
                res = (lhs > rhs);
                break;

            case CMP_GR:
                res = (lhs >= rhs);
                break;

            default:
                GECODE_NEVER
        }
        if (!res) return false;
    }

    for (auto it = mCstrLinear.begin(); it != mCstrLinear.end(); ++it) {
        int lhs = 0;
        int rhs = vars[it->v0];
        bool res;

        for (auto jt = it->poly.begin(); jt != it->poly.end(); ++jt) {
            lhs += jt->first * jt->second;
        }

        switch (it->cmp) {
            case CMP_NQ:
                res = (lhs != rhs);
                break;

            case CMP_EQ:
                res = (lhs == rhs);
                break;

            case CMP_LQ:
                res = (lhs < rhs);
                break;

            case CMP_LE:
                res = (lhs <= rhs);
                break;

            case CMP_GQ:
                res = (lhs > rhs);
                break;

            case CMP_GR:
                res = (lhs >= rhs);
                break;

            default:
                GECODE_NEVER
        }
        if (!res) return false;
    }

    return true;
}
