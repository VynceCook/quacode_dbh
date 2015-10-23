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

TestAlgo::TestAlgo() : AsyncAlgo() {
    mNbVars = 0;
}

TestAlgo::~TestAlgo() {
    mbQuacodeThreadFinished = true;

    mDestructor.acquire();
    mDestructor.release();
}

void TestAlgo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    OSTREAM << "New var " << idx << "/" <<  printType(t) << " : " << name << ", type " << (q == EXISTS ? "E" : "F") << ", dom {" << min << "," << max << "}" <<std::endl;
    mVars.push_back({q, name, t, min, max});
}

void TestAlgo::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    OSTREAM << "New auxiliary var " << name << "/" << printType(t) << ", dom {" << min << "," << max << "}" << std::endl;
    mVarsAux.push_back({0, name, t, min, max});
}

void TestAlgo::newChoice(int iVar, int min, int max) {
    OSTREAM << "Chef ! We need to explore others choices : " << iVar << " {" << min << "," << max << "}" << std::endl;
}
void TestAlgo::newPromisingScenario(const TScenario& scenario) {
    OSTREAM << "Chef ! I think this scenario is interesting : ";
    for (auto s: scenario) {
        OSTREAM << "{" << s.min << "," << s.max << "}, ";
    }
    OSTREAM << std::endl;
}
void TestAlgo::strategyFound() {
    OSTREAM << "Chef ! We just found a solution !" << std::endl;
}
void TestAlgo::newFailure() {
    OSTREAM << "Another faillure chef !" << std::endl;
}

void TestAlgo::globalFailure() {
    OSTREAM << "It can't be done. ABORT ! ABORT ! ABORT !" << std::endl;
}

/// Function called when a new 'v0 == v2' constraint is posted
void TestAlgo::postedEq(const std::string& v0, int val) {
    OSTREAM << "New constraint Eq " << v0 << "=" << val << std::endl;
}

/// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint And " << p0 << ":" << v0 << " && " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}

/// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint Or " << p0 << ":" << v0 << " || " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}

/// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint Imp " << p0 << ":" << v0 << " >> " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}

/// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void TestAlgo::postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint Xor " << p0 << ":" << v0 << " ^ " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}


/// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
void TestAlgo::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint Plus " << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}

/// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
void TestAlgo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    OSTREAM << "New constraint Times " << n << "*" << v0 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;
}

/// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
void TestAlgo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    OSTREAM << "New constraint Linear ";
    for (auto m : poly) {
        OSTREAM << m.coeff << "*" << m.varName << " + ";
    }
    OSTREAM << printComp(cmp) << " " << v0 << std::endl;
}


void TestAlgo::parallelTask() {
    OSTREAM << "THREAD start" << std::endl;

    mDestructor.acquire();
    srand(time(NULL));
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;

        Gecode::Support::Thread::sleep(300);
    }

    mDestructor.release();
    OSTREAM << "THREAD stop" << std::endl;
}
