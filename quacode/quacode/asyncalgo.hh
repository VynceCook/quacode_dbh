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
#include <queue>

#include <gecode/support.hh>
#include <gecode/int.hh>
#include <quacode/qcsp.hh>

// Forward declaration
namespace Gecode { namespace Int { namespace Branch {
    template<int n, bool min> class QViewValuesOrderBrancher;
    class QPosValuesOrderChoice;
}}}

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

// Structure which represents an interval
struct Interval {
    int min;
    int max;
};
typedef std::vector<Interval> TScenario;

// Structure which represents a monome in a polynome
struct Monom {
    int coeff;
    std::string varName;
};

// Structure which represents the description of a variable
struct TVarDesc {
    Gecode::TQuantifier q;
    std::string name;
    TVarType type;
    int min;
    int max;
};

class AsyncAlgo : public Gecode::Support::Runnable {
    template<int n, bool min> friend class Gecode::Int::Branch::QViewValuesOrderBrancher;
    friend class Gecode::Int::Branch::QPosValuesOrderChoice;

    /// Copy constructor set private to disable it.
    AsyncAlgo(const AsyncAlgo&);

    /// Stores the ordered domain of each variable of the binder
    std::vector< std::vector<int> > mDomains;
    /// Mutex for access to mDomains
    mutable std::vector< Gecode::Support::Mutex* > mDomainsMutex;

    /// Wrapper function executed when the thread starts
    QUACODE_EXPORT virtual void run(void);

    /// Copy the domain of the given variable \a iVar to dest.
    /// It copies only the elemnts that are in the IntView \a iv
    void copyDomainIf(int iVar, const Gecode::Int::IntView& iv, std::vector<int>& dest) const;
public:
    /// Main constructor
    QUACODE_EXPORT AsyncAlgo();

    // Main destructor
    QUACODE_EXPORT virtual ~AsyncAlgo();

    ///-----------------------------------------------------------------------
    /// ------- These functions are called until the 'closeModeling' function is called
    ///-----------------------------------------------------------------------

    /// ====== These are to be called during modeling stage
    /// ===================================================
    /// Quacode adds a new variable \a var in the binder
    QUACODE_EXPORT void newVar(Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max);
    /// Quacode adds a new variable auxiliary \a var
    QUACODE_EXPORT void newAuxVar(const std::string& name, TVarType t, int min, int max);

    /// Quacode post a new v0 == v2
    QUACODE_EXPORT void postEq(std::string v0, int val);

    /// Quacode post a new p0v0 && p1v1 <cmp> v2  (p0, p1 are polarity of literals)
    QUACODE_EXPORT void postAnd(bool p0, std::string v0, bool p1, std::string v1, TComparisonType cmp, std::string v2);
    /// Quacode post a new p0v0 || p1v1 <cmp> v2  (p0, p1 are polarity of literals)
    QUACODE_EXPORT void postOr(bool p0, std::string v0, bool p1, std::string v1, TComparisonType cmp, std::string v2);
    /// Quacode post a new p0v0 >> p1v1 <cmp> v2  (p0, p1 are polarity of literals)
    QUACODE_EXPORT void postImp(bool p0, std::string v0, bool p1, std::string v1, TComparisonType cmp, std::string v2);
    /// Quacode post a new p0v0 ^ p1v1 <cmp> v2  (p0, p1 are polarity of literals)
    QUACODE_EXPORT void postXOr(bool p0, std::string v0, bool p1, std::string v1, TComparisonType cmp, std::string v2);

    /// Quacode post a new n0*v0 + n1*v1 <cmp> v2
    QUACODE_EXPORT void postPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2);
    /// Quacode post a new n*v0*v1 <cmp> v2
    QUACODE_EXPORT void postTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2);
    /// Quacode post a new SUM_i n_i*v_i <cmp> v0
    QUACODE_EXPORT void postLinear(const std::vector<Monom>& poly, TComparisonType cmp, std::string v0);

    /// ====== These are to be redefined in subclass
    /// ====== as they are automatically called by the previous ones
    /// ===================================================
    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    QUACODE_EXPORT virtual void newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) = 0;
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    QUACODE_EXPORT virtual void newAuxVarCreated(const std::string& name, TVarType t, int min, int max) = 0;

    /// Function called when a new 'v0 == v2' constraint is posted
    QUACODE_EXPORT virtual void postedEq(const std::string& v0, int val);

    /// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    QUACODE_EXPORT virtual void postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    QUACODE_EXPORT virtual void postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    QUACODE_EXPORT virtual void postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    QUACODE_EXPORT virtual void postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);

    /// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
    QUACODE_EXPORT virtual void postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    QUACODE_EXPORT virtual void postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    QUACODE_EXPORT virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0);

    /// ====== The closeModeling function must be called at the
    /// ====== end of the modeling stage. It will ends the modeling
    /// ====== and start another thread with the parallel algorithm
    /// ===================================================
    /// Quacode closes the modeling step.
    QUACODE_EXPORT void closeModeling();

    ///-----------------------------------------------------------------------
    /// ------- These functions are called after the 'closeModeling' function is called
    ///-----------------------------------------------------------------------

    /// ====== These are to be redefined in subclass
    /// ====== as they are automatically called during search
    /// ===================================================
    /// Function called when a new choice (\a iVar = variable index in the binder,
    /// \a min and \a max are the lower and upper bounds of the value) during search
    QUACODE_EXPORT virtual void newChoice(int iVar, int min, int max) = 0;
    /// Function called when a new promising scenario is discovered during search
    QUACODE_EXPORT virtual void newPromisingScenario(const TScenario& instance) = 0;
    /// Function called when the search ends with a successfull strategy
    QUACODE_EXPORT virtual void strategyFound() = 0;
    /// Function called when a failure occured during search
    QUACODE_EXPORT virtual void newFailure() = 0;
    /// Function called when the search ends with a global failure, problem unfeasible
    QUACODE_EXPORT virtual void globalFailure() = 0;

    ///-----------------------------------------------------------------------
    /// ------- These functions are thread safe and can be called during the
    /// ------- search by the parallel algorithm
    ///-----------------------------------------------------------------------
    /// Ask a swap of two values \a iV0 and \a iV1 of variable \a iVar
    QUACODE_EXPORT void swap(unsigned int iVar, unsigned int iV0, unsigned int iV1);
    /// Returns the \a nth value of domain of the given variable \a iVar
    QUACODE_EXPORT int getValue(int iVar, int nth) const;
    /// Copy the domain of the given variable \a iVar to dest
    QUACODE_EXPORT void copyDomain(int iVar, std::vector<int>& dest) const;

    ///-----------------------------------------------------------------------
    /// ------- This is the main function called in another thread launched.
    /// ------- It has to be redefined, it is the place to put the code of the
    /// ------- parallel search algorithm.
    ///-----------------------------------------------------------------------
    /// Function executed when the thread starts
    QUACODE_EXPORT virtual void parallelTask(void) = 0;
};

#include <quacode/asyncalgo/asyncalgo.hpp>

#endif
