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

bool GenAlgo::Constraint::cmpEQ(int lhs, int rhs) {
    return (lhs == rhs);
}

bool GenAlgo::Constraint::cmpNQ(int lhs, int rhs) {
    return (lhs != rhs);
}

bool GenAlgo::Constraint::cmpGQ(int lhs, int rhs) {
    return (lhs > rhs);
}

bool GenAlgo::Constraint::cmpGR(int lhs, int rhs) {
    return (lhs >= rhs);
}

bool GenAlgo::Constraint::cmpLQ(int lhs, int rhs) {
    return (lhs < rhs);
}

bool GenAlgo::Constraint::cmpLE(int lhs, int rhs) {
    return (lhs <= rhs);
}

GenAlgo::Constraint::cmpFuncPtr GenAlgo::Constraint::getCmpPtr(TComparisonType t) {
    switch (t) {
        case CMP_NQ:
            return &GenAlgo::Constraint::cmpNQ;
        case CMP_EQ:
            return &GenAlgo::Constraint::cmpEQ;
        case CMP_LQ:
            return &GenAlgo::Constraint::cmpLQ;
        case CMP_LE:
            return &GenAlgo::Constraint::cmpLE;
        case CMP_GQ:
            return &GenAlgo::Constraint::cmpGQ;
        case CMP_GR:
            return &GenAlgo::Constraint::cmpGR;
    }

    return nullptr;
}


GenAlgo::CstrEq::CstrEq(size_t v0, int v2):
    v0(v0), v2(v2)
{}

bool GenAlgo::CstrEq::evaluate(const std::vector<int> & c) {
    return c[v0] == c[v2];
}

GenAlgo::CstrBool::CstrBool(bool p0, size_t v0, opFuncPtr op, bool p1, size_t v1, cmpFuncPtr cmp, size_t v2):
    p0(p0), v0(v0), op(op), p1(p1), v1(v1), cmp(cmp), v2(v2)
{}

bool GenAlgo::CstrBool::evaluate(const std::vector<int> & c) {
    return cmp(op(p0, c[v0], p1, c[v1]), c[v2]);
}

bool GenAlgo::CstrBool::opAnd(bool p0, bool v0, bool p1, bool v1) {
    return (p0 ? v0 : !v0) && (p1 ? v1 : !v1);
}

bool GenAlgo::CstrBool::opOr(bool p0, bool v0, bool p1, bool v1) {
    return (p0 ? v0 : !v0) || (p1 ? v1 : !v1);
}

bool GenAlgo::CstrBool::opImp(bool p0, bool v0, bool p1, bool v1) {
    return !((p0 ? v0 : !v0) && !(p1 ? v1 : !v1));
}

bool GenAlgo::CstrBool::opXor(bool p0, bool v0, bool p1, bool v1) {
    return (!(p0 ? v0 : !v0) != !(p1 ? v1 : !v1));
}

GenAlgo::CstrPlus::CstrPlus(int n0, size_t v0, int n1, size_t v1, cmpFuncPtr cmp, size_t v2):
    n0(n0), v0(v0), n1(n1), v1(v1), cmp(cmp), v2(v2)
{}

bool GenAlgo::CstrPlus::evaluate(const std::vector<int> & c) {
    return cmp(n0 * c[v0] + n1 * c[v1], v2);
}

GenAlgo::CstrTimes::CstrTimes(int n, size_t v0, size_t v1, cmpFuncPtr cmp, size_t v2):
    n(n), v0(v0), v1(v1), cmp(cmp), v2(v2)
{}

bool GenAlgo::CstrTimes::evaluate(const std::vector<int> & c) {
    return cmp(n * c[v0] * c[v1], v2);
}

GenAlgo::CstrLinear::CstrLinear(const std::vector<std::pair<int, size_t>> &poly, cmpFuncPtr cmp, size_t v0):
    poly(poly), cmp(cmp), v0(v0)
{}

bool GenAlgo::CstrLinear::evaluate(const std::vector<int> & c) {
    int sum = 0;

    for (auto it = poly.begin(); it != poly.end(); ++it) {
        sum += it->first * c[it->second];
    }

    return cmp(sum, c[v0]);
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

    for (auto it = mCstrs.begin(); it != mCstrs.end(); ++it) {
        delete *it;
    }

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
        Constraint * tmp = new CstrEq(v0Idx, val);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrBool(p0, v0Idx, &CstrBool::opAnd, p1, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrBool(p0, v0Idx, &CstrBool::opOr, p1, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrBool(p0, v0Idx, &CstrBool::opImp, p1, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrBool(p0, v0Idx, &CstrBool::opXor, p1, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrPlus(n0, v0Idx, n1, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrTimes(n, v0Idx, v1Idx, Constraint::getCmpPtr(cmp), v2Idx);
        mCstrs.push_back(tmp);
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
        Constraint * tmp = new CstrLinear({}, Constraint::getCmpPtr(cmp), v0Idx);
        mCstrs.push_back(tmp);

        for (auto it = poly.begin(); it != poly.end(); ++it) {
            size_t viIdx = findVar(it->varName);

            if (viIdx != (size_t)-1) {
                dynamic_cast<CstrLinear*>(tmp)->poly.push_back(std::make_pair(it->coeff, viIdx));
            }
            else {
                OSTREAM << "Can't find " << it->varName << std::endl;
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
    for (auto it = mCstrs.begin(); it != mCstrs.end(); ++it) {
        if (!(*it)->evaluate(vars)) return false;
    }

    return true;
}


std::vector<std::vector<int>> GenAlgo::generateAll() {
	std::vector<std::vector<int>> res;
	std::vector<int> cur(mNbVars);
	int i;

	while (1) {
		for (i=0; cur[i] == mVars[i].dom.max && i < mNbVars; ++i) {
			cur[i] = mVars[i].dom.min;
		}

		if (i > mNbVars) {
			// We generated all the possible var sets
			break;
		}
		cur[i] ++;

		if (evaluate(cur)) {
			res.push_back(std::vector<int>(cur));
		}
	}

	return(res);
}


std::vector<std::vector<int>> GenAlgo::generateRandom(const int n) {
	std::vector<std::vector<int>> res(n);

	srand(time(NULL));

	// generates random sets of var (useful in genetic algo's initialisation)
	for (int i=0; i<n; ++i) {
		res[i] = std::vector<int>(mNbVars);
		for (int j=0; j<mNbVars; ++j) {
			res[i][j] = mVars[j].dom.min + (rand() % (mVars[j].dom.max - mVars[j].dom.min));
		}
	}

	return(res);
}
