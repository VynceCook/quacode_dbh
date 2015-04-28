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

#include <algorithms/logger.hh>
#define OSTREAM std::cerr

Logger::Logger() : AsyncAlgo(), mbQuacodeThreadFinished(false) { }

Logger::~Logger() {
    mbQuacodeThreadFinished = true;
    // We block the destructor until the background thread finished
    mDestructor.acquire();
    mDestructor.release();
    OSTREAM << "OBJECT DESTROYED" << std::endl;
}

void Logger::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    mVarNames.push_back(name);
    OSTREAM << "VAR_BINDER       =";
    OSTREAM << " var(" << ((q==EXISTS)?"E":"F") << "," << ((t==TYPE_BOOL)?"B":"I") << "," << name << " # " << idx;
    if (min == max)
        OSTREAM << "," << min;
    else
        OSTREAM << ",[" << min << ":" << max << "]";
    OSTREAM << ")" << std::endl;
}

void Logger::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    OSTREAM << "VAR_AUX          =";
    OSTREAM << " var(E," << ((t==TYPE_BOOL)?"B":"I") << "," << name;
    if (min == max)
        OSTREAM << "," << min;
    else
        OSTREAM << ",[" << min << ":" << max << "]";
    OSTREAM << ")" << std::endl;
}

void Logger::newChoice(int iVar, int min, int max) {
    OSTREAM << "CHOICE           = ";
    if (min == max)
        OSTREAM << mVarNames[iVar] << " # " << min << std::endl;
    else
        OSTREAM << mVarNames[iVar] << " # [" << min << ";" << max << "]" << std::endl;
}
void Logger::newPromisingScenario(const TScenario& scenario) {
    bool bFirst = true;
    OSTREAM << "PR SCENARIO      = ";
    for(const auto &iv : scenario) {
        if (!bFirst) OSTREAM << ", ";
        if (iv.min == iv.max)
            OSTREAM << iv.min;
        else
            OSTREAM << "[" << iv.min << ";" << iv.max << "]";
        bFirst = false;
    }
    OSTREAM << std::endl;
}
void Logger::strategyFound() {
    OSTREAM << "STRATEGY FOUND" << std::endl;
}
void Logger::newFailure() {
    OSTREAM << "FAIL" << std::endl;
}
void Logger::globalFailure() {
    OSTREAM << "GLOBAL FAILURE" << std::endl;
}

void Logger::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    OSTREAM << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void Logger::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    if (n != 1) OSTREAM << n << " * ";
    OSTREAM << v0 << " * " << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void Logger::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    bool bFirst = true;
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    for(const auto &m : poly) {
        if (!bFirst) OSTREAM << " + ";
        OSTREAM << m.coeff << "*" << m.varName;
        bFirst = false;
    }
    OSTREAM << " " << s_ComparisonType[cmp] << " " << v0 << std::endl;
}

void Logger::parallelTask() {
    // If mainThread has finished, member variables are not usable anymore
    // so we block the destructor until we finish here
    mDestructor.acquire();

    OSTREAM << "THREAD start" << std::endl;
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;
        OSTREAM << "THREAD ..." << std::endl;
        Gecode::Support::Thread::sleep(300);
    }
    OSTREAM << "THREAD stop" << std::endl;
    // Release the destructor as we have finish everything
    mDestructor.release();
}
