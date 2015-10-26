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

#ifndef __TEST_H__
#define __TEST_H__

#include <quacode/asyncalgo.hh>

class TestAlgo : public AsyncAlgo {
    /// Variables of the problem
    struct VarDesc {
        int idxInBinder;
        Gecode::TQuantifier q;
        std::string name;
        TVarType type;
        Interval dom;
        Interval curDom;
    };

    /// Copy constructor set private to disable it.
    TestAlgo(const TestAlgo&);

    /// Flag to know if the main thread finished its work
    bool mbQuacodeThreadFinished;
    /// Mutex to block the destructor
    Gecode::Support::Mutex mDestructor;
    /// Stores the number of variables of the binder
    int mNbVars;
    int mNbBinderVars;
    int mLastChoice;

    std::vector<VarDesc> mVars;

    // Restaure the domain of var in the interval ]from, to]
    void restaureDomaines(int from, int to);

public:
    /// Main constructor
    TestAlgo();

    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    virtual void newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max);
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    virtual void newAuxVarCreated(const std::string& name, TVarType t, int min, int max);

    /// Function called when a new choice (\a iVar = variable index in the binder,
    /// \a min and \a max are the lower and upper bounds of the value) during search
    virtual void newChoice(int iVar, int min, int max);
    /// Function called when a new promising scenario is discovered during search
    virtual void newPromisingScenario(const TScenario& instance);
    /// Function called when the search ends with a successfull strategy
    virtual void strategyFound();
    /// Function called when a failure occured during search
    virtual void newFailure();
    /// Function called when the search ends with a global failure, problem unfeasible
    virtual void globalFailure();

    /// Function called when a new 'v0 == v2' constraint is posted
    virtual void postedEq(const std::string& v0, int val);

    /// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);

    /// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
    virtual void postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    virtual void postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0);

    /// Function executed when the thread starts
    virtual void parallelTask(void);

    // Main destructor
    virtual ~TestAlgo();
};

#endif
