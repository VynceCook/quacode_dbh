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

#include <asyncalgo/asyncalgo.hh>

// // Initialisation of instance of singleton
// SIBus* SIBus::m_pSIBusInstance = NULL;
// 
// void Receiver::setSIBus(SIBus const * _sibus) {
//   sibus = _sibus;
// }
// 
// Receiver::~Receiver() { }
// 
// SIBus::SIBus() : sibusState(S_INIT), binderSize(0) {
// #ifdef SIBUS_THREAD
//   m_thread = NULL;
// #endif
// }
// 
// void SIBus::create() {
//   assert(m_pSIBusInstance == NULL);
//   m_pSIBusInstance = new SIBus();
// 
// #ifdef SIBUS_THREAD
//   m_pSIBusInstance->m_thread = new boost::thread(&SIBus::run,m_pSIBusInstance);
// #endif
// }
// 
// void SIBus::kill() {
//   assert(m_pSIBusInstance != NULL);
//   delete m_pSIBusInstance;
//   m_pSIBusInstance = NULL;
// }
// 
// void SIBus::run(void) {
// #ifdef SIBUS_THREAD
//   // Start working
//   for ( ; ; ) {
//     mx.acquire();
//     while (!m_fifo.empty()) {
//       manageEvent(m_fifo.front());
//       m_fifo.pop_front();
//       mx.release();
//       mx.acquire();
//     }
//     if (sibusState == S_SHUTDOWN) break;
//     mx.release();
//     ev_fifo.wait();
//   }
//   mx.release();
// #endif
// }

#define OSTREAM std::cerr

AsyncAlgo::AsyncAlgo() { }

AsyncAlgo::~AsyncAlgo() {
    OSTREAM << "END" << std::endl;
#ifdef USE_THREAD
    //   mx.acquire();
#endif
    //   sibusState = S_SHUTDOWN;
#ifdef USE_THREAD
    //   ev_fifo.signal();
    //   mx.release();
    //   m_thread->join();
    //   delete m_thread;
    //   m_thread = NULL;
    //   sibusState = S_OFF;
#endif
}

void AsyncAlgo::closeModeling() {
    OSTREAM << "CLOSE MODELING" << std::endl;
}

void AsyncAlgo::newVar(Gecode::TQuantifier q, std::string name, TVarType t, TVal v) {
    OSTREAM << "VAR_BINDER       =";
    OSTREAM << " var(" << ((q==EXISTS)?"E":"F") << "," << ((t==TYPE_BOOL)?"B":"I") << "," << name;
    switch (v.type) {
        case VAL_NONE:
            break;
        case VAL_BOOL:
            OSTREAM << ",bool(" << v.val.b << ")";
            break;
        case VAL_INT:
            OSTREAM << ",int(" << v.val.z << ")";
            break;
        case VAL_INTERVAL:
            OSTREAM << ",interval(" << v.val.bounds[0] << ":" << v.val.bounds[1] << ")";
            break;
    }
    OSTREAM << ")" << std::endl;
}

// Add a new auxiliary variable \a var
void AsyncAlgo::newAuxVar(std::string name, TVarType t, TVal v) {
    OSTREAM << "VAR_AUX          =";
    OSTREAM << " var(E," << ((t==TYPE_BOOL)?"B":"I") << "," << name;
    switch (v.type) {
        case VAL_NONE:
            break;
        case VAL_BOOL:
            OSTREAM << ",bool(" << v.val.b << ")";
            break;
        case VAL_INT:
            OSTREAM << ",int(" << v.val.z << ")";
            break;
        case VAL_INTERVAL:
            OSTREAM << ",interval(" << v.val.bounds[0] << ":" << v.val.bounds[1] << ")";
            break;
    }
    OSTREAM << ")" << std::endl;
}

void AsyncAlgo::newChoice(int idx, TVal val) {
    OSTREAM << "CHOICE           = ";
    if (val.type == VAL_INTERVAL)
        OSTREAM << idx << " # [" << val.val.bounds[0] << ";" << val.val.bounds[1] << "]" << std::endl;
    else if (val.type == VAL_BOOL)
        OSTREAM << idx << " # " << val.val.b << std::endl;
    else
        OSTREAM << idx << " # " << val.val.z << std::endl;
}
void AsyncAlgo::newPromisingScenario(const TScenario& scenario) {
    bool bFirst = true;
    OSTREAM << "PR SCENARIO      = ";
    for(auto &v : scenario) {
        if (!bFirst) OSTREAM << ", ";
        if (v.type == VAL_NONE)
            OSTREAM << "NOVALUE)";
        else if (v.type == VAL_INTERVAL)
            OSTREAM << "[" << v.val.bounds[0] << ";" << v.val.bounds[1] << "]";
        else if (v.type == VAL_BOOL)
            OSTREAM << v.val.b;
        else
            OSTREAM << v.val.z;
        bFirst = false;
    }
    OSTREAM << std::endl;
}
void AsyncAlgo::strategyFound() {
    OSTREAM << "STRATEGY FOUND" << std::endl;
}
void AsyncAlgo::newFailure() {
    OSTREAM << "FAIL" << std::endl;
}
void AsyncAlgo::globalFailure() {
    OSTREAM << "GLOBAL FAILURE" << std::endl;
}

void AsyncAlgo::postPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    OSTREAM << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void AsyncAlgo::postTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2) {
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    if (n != 1) OSTREAM << n << " * ";
    OSTREAM << v0 << " * " << v1 << " " << s_ComparisonType[cmp] << " " << v2 << std::endl;
}

void AsyncAlgo::postLinear(std::vector<Monom> poly, TComparisonType cmp, std::string v0) {
    bool bFirst = true;
    static char s_ComparisonType[][20] = { "!=", "==", "<", "<=", ">", ">=" };
    OSTREAM << "POST             = ";
    for(auto &m : poly) {
        if (!bFirst) OSTREAM << " + ";
        OSTREAM << m.c << "*" << m.var;
        bFirst = false;
    }
    OSTREAM << " " << s_ComparisonType[cmp] << " " << v0 << std::endl;
}
