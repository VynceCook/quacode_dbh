/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *    Vincent Barichard, 2015
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

#ifndef __ASYNCALGO_HH__
#define __ASYNCALGO_HH__

#include <cassert>
#include <vector>
#include <string>
#include <list>

#ifdef USE_THREAD
  #include <gecode/support/thread.hpp>
#endif

#include <quacode/qcsp.hh>

#define TYPE_BOOL  0
#define TYPE_INT   1
// Information on type of a variable
typedef unsigned int TVarType;

#define CMP_NQ  0
#define CMP_EQ  1
#define CMP_LQ  2
#define CMP_LE  3
#define CMP_GQ  4
#define CMP_GR  5
// Information on comparison operator
typedef unsigned int TComparisonType;

// Availables values types for a value.
// Either a value (bool or int), or an interval
#define VAL_NONE      0
#define VAL_BOOL      1
#define VAL_INT       2
#define VAL_INTERVAL  3
// Information on type of argument of a constraint
typedef unsigned int TValType;

// Availables values types for a variable.
struct TVal {
    TValType type;
    union {
        bool b;
        int  z;
        int  bounds[2];
    } val;
    TVal() : type(VAL_NONE) {}
    TVal(bool _b) : type(VAL_BOOL) { val.b = _b; }
    TVal(int  _z) : type(VAL_INT) { val.z = _z; }
    TVal(int _min, int _max) : type(VAL_INTERVAL) { val.bounds[0] = _min; val.bounds[1] = _max; }

    bool operator!=(const TVal& v) const { return !this->operator==(v); }
    bool operator==(const TVal& v) const {
        return ((v.type == type) && (
                    (type == VAL_NONE) ||
                    ((type == VAL_BOOL) && (val.b == v.val.b)) ||
                    ((type == VAL_INT) && (val.z == v.val.z)) ||
                    ((type == VAL_INTERVAL) && (val.bounds[0] == v.val.bounds[0]) && (val.bounds[1] == v.val.bounds[1]))
                    ));
    }
};
typedef std::vector<TVal> TScenario;

struct Monom {
    int c;
    std::string var;
};

class AsyncAlgo {
private:
//#ifdef SIBUS_THREAD
//    // Thread
//    boost::thread * m_thread;
//    // Mutex for access to sibus
//    mutable BoostMutex mx;
//    /// Event for push data in queue
//    mutable BoostEvent ev_fifo;
//    // List of events to be sent to receivers
//    mutable std::list<TLEvent> m_fifo;
//#endif
//
//    // States of the SIBus (init, run, shutdown, off)
//    static const unsigned int S_INIT     = 0;
//    static const unsigned int S_RUN      = 1;
//    static const unsigned int S_SHUTDOWN = 2;
//    static const unsigned int S_OFF      = 3;
//    unsigned int sibusState;
//
//    // Size of an instance (number of variables of the instance)
//    unsigned binderSize;
//
    // Copy constructor set private to disable it.
    AsyncAlgo(const AsyncAlgo&);

public:
    // Main constructor set private to disable it.
    AsyncAlgo();

    // A new variable \a var has been added as a part of the binder
    virtual void newVar(Gecode::TQuantifier q, std::string name, TVarType t, TVal v);
    // A new auxiliary variable \a var has been added
    virtual void newAuxVar(std::string name, TVarType t, TVal v);

    // Close the modeling step.
    virtual void closeModeling();

    // Add a new choice of the search tree to the m_fifo data structure. \a idx gives
    // the index of the variable in the binder, and \a val the chosen value
    virtual void newChoice(int idx, TVal val);
    // Discovered a new promising scenario during search
    virtual void newPromisingScenario(const TScenario& instance);
    // Ends the search with a successfull strategy
    virtual void strategyFound();
    // A failure occured
    virtual void newFailure();
    // Ends the search with a global failure, problem unfeasible
    virtual void globalFailure();

    // Post a new n0*v0 + n1*v1 <cmp> v2
    virtual void postPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2);
    // Post a new n*v0*v1 <cmp> v2
    virtual void postTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2);
    // Post a new SUM_i n_i*v_i <cmp> v0
    virtual void postLinear(std::vector<Monom> poly, TComparisonType cmp, std::string v0);

//      // Ask a swap of two values \a idVal1 and \a idVal2 of variable \a idVar
//      virtual void sendSwapAsk(unsigned int idVar, unsigned int idVal1, unsigned int idVal2);
//
//      // Send a swap done
//      virtual void sendSwapDone(unsigned int idVar, unsigned int idVal1, unsigned int idVal2);
//
      // Main destructor
      virtual ~AsyncAlgo();
};

#include <asyncalgo/asyncalgo.hpp>

#endif
